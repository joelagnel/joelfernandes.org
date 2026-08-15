---
layout: post
title: "llama.cpp and the ubatch"
date: 2026-08-14
categories: [machine-learning, inference]
tags: [machine-learning, ml, llama-cpp, inference, transformers, attention, continuous-batching, kv-cache, vram, ubatch]
author: Joel Fernandes
description: "Notes from working out why -ub changes VRAM when only one session is active, why the ubatch is not the batch, and what continuous batching actually does inside a transformer layer."
published: true
---

I spent some time recently trying to fit a model into a fixed amount of VRAM,
and hit an interesting puzzle. The suggestion was to lower `--ubatch-size`. But
I only ever had one session talking to the server, and during generation a
single session produces one token at a time. On the face of it, whatever that
knob controlled was already one, and lowering one to something smaller should do
nothing. So why `--ubatch-size`?

Digging into it, I learned that the u stands for micro, and that the word batch
means three different things depending on who is saying it. Once those three
came apart, the knob made sense.

## Three things called batch

Most of my intuition about the word batch comes from training. There the
activations flowing through the model have shape `[B, T, D]`. `B` is the batch
size, the number of independent training samples being processed together. `T`
is the time dimension, the token positions within a sample. `D` is the embedding
dimension, the vector each individual token is represented by.

So if I train on four sentences at once, each 128 tokens long, with a model whose
embedding width is 4096, the tensor is `[4, 128, 4096]`. Sample 0 might be a
paragraph about cooking and sample 3 a snippet of Python. They are unrelated;
they ride along together only because the GPU is happier with one large matmul
than four small ones. Each of the four samples holds 128 positions, and at each
position sits one 4096-dimensional vector.

That is meaning one, and the key property is that `B` counts *independent*
things. Row 0 and row 3 have nothing to say to each other.

Meaning two is llama.cpp's batch, which is a list of token slots handed to one
`llama_decode()` call. Each slot carries a token, a position, and the sequence
it belongs to. Six slots means six token positions, which might be six positions
from one conversation, or one position each from six conversations, or any
mixture. It is a work order, not a stack of examples.

<div style="margin: 1.5em 0;">
  <img src="/images/ubatch/batch-two-meanings.svg"
       alt="Left: a training batch of four rows, each row a separate example, annotated as the batch dimension, shape [4, 128, 4096]. Right: a llama.cpp ubatch of four rows, each row a single token slot tagged with a sequence id and a position, with rows belonging to three different conversations."
       style="width: 100%; height: auto; display: block;"/>
</div>

The naming collision between those first two is not worth dwelling on, but the
picture is, since almost everything below follows from the fact that
llama.cpp's rows are token slots rather than independent examples.

The actual struct in `include/llama.h` is a little more general than I drew it.
Each slot has an `n_seq_id` count and a *list* of sequence ids, not a single one,
because a token can belong to more than one sequence at a time. That is how
prefix sharing works: a common prompt prefix lives in the cache once and several
sequences point at it.

Meaning three is the ubatch, and it is the one this post is really about. The
four knobs in this area are worth separating first:

- `-np` limits how many conversations can be alive at once. This is the one
  closest to the training meaning of batch.
- `-c` sets how many token positions the KV cache can hold.
- `-b` sets the largest list of token slots you are allowed to hand to a single
  `llama_decode()` call. This is meaning two.
- `-ub` sets how many token positions go through the network in one forward
  pass. This is meaning three.

The last two are the pair worth pinning down, and the header comments in
`include/llama.h` distinguish them neatly:

```c
uint32_t n_batch;   // logical maximum batch size that can be submitted to llama_decode
uint32_t n_ubatch;  // physical maximum batch size
```

Logical means at the API boundary. `n_batch` is a limit on the caller. If I hand
`llama_decode()` more slots than that, it refuses:

```cpp
GGML_ASSERT(n_tokens_all <= cparams.n_batch);
```

Physical means inside the engine. `n_ubatch` is what llama.cpp does with the
list it was given: it chops it into pieces of at most `n_ubatch` slots, and each
piece is a ubatch, one trip through the network. So with the defaults of
`n_batch = 2048` and `n_ubatch = 512`, a 5000-token prompt becomes three
`llama_decode()` calls from the caller's side (2048, 2048, 904), and each of
those is split again into ubatches of at most 512, which are what the GPU
actually sees. Ten forward passes in total: four, four, then two.

A couple of consequences fall out of that. `n_ubatch` is clamped to `n_batch` at
context creation, so a ubatch can never be larger than the batch it came from.
And under causal attention `n_batch` is itself clamped to `n_ctx`, since you
cannot submit more positions than the cache can hold:

```cpp
cparams.n_batch  = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;
cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);
```

Raising `-b` on its own therefore does not change the shape of the work the GPU
does. It lets the caller submit longer lists, which saves some per-call
overhead, but the tensor shapes in each forward pass come from `-ub`.

So the batch is a request and the ubatch is what actually runs. For the rest of
this post I use them in exactly that sense: batch means the list submitted at
the API boundary, ubatch means the group of token slots that goes through the
network together. The training meaning does not come back. Everything
interesting below, the mixing of conversations, the masking, the row-wise
argument, happens at the ubatch level, because that is the only one of the three
that corresponds to a real tensor.

## Where the memory actually sits

Two words are about to do a lot of work, so it is worth pinning them down.

The **KV cache** is the memory of the conversation. Attention works by comparing
the current token's query against a key for every earlier token, and then mixing
together those earlier tokens' values. Keys and values, hence KV. They depend
only on the token and its position, so once a token has been through a layer its
key and value never change. Recomputing them for the whole prefix on every new
token would be quadratic work, so llama.cpp stores them. That store is the KV
cache, and its size grows with how many positions you allow (`-c`) and how many
bytes each entry takes.

The **graph** is the computation itself, written down. Before running anything,
llama.cpp builds a ggml compute graph: a list of tensor operations with fixed
shapes, wired input to output, describing one forward pass. Nothing has run yet;
it is a plan. Because the shapes are fixed at build time, the allocator can walk
the graph and work out exactly how much scratch memory the intermediate tensors
need, and it can reuse the same buffers for tensors whose lifetimes do not
overlap. That is why sizes have to be known up front, and it is also why a graph
whose shapes do not change can be built once and reused.

With that, VRAM during inference splits roughly three ways: the weights, the KV
cache, and a workspace for the graph's intermediate tensors. Each knob moves a
different piece.

<div style="margin: 1.5em 0;">
  <img src="/images/ubatch/vram-slabs.svg"
       alt="Two stacked bars showing VRAM split into model weights, KV cache, and compute workspace. The right bar has the same weights and KV cache but a much smaller compute workspace, labelled as the effect of lowering ubatch from 512 to 128."
       style="width: 100%; height: auto; display: block;"/>
</div>

`-c` and the KV cache precision set the middle slab. `-ub` sets the top one. The
weights do not care about either.

The reason `-ub` matters even with a single session is that llama.cpp sizes the
workspace once, up front, for the largest forward pass it might ever be asked to
run. In `llama_context::sched_reserve()`:

```cpp
const uint32_t n_seqs   = cparams.n_seq_max;
const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
```

It reserves against the prompt-processing graph, then the token-generation
graph, then the prompt-processing graph again, so that nothing has to be
reallocated later when switching between the two. That worst case is driven by
`n_ubatch`, which follows from the split: the largest tensor the graph ever holds
is one ubatch wide, no matter how long a list the caller submitted.

`n_batch` is not entirely free, but what it sizes is the output side rather than
the compute workspace. In `output_reserve()` the `output_ids` map is `n_batch`
entries, and the optional per-layer embedding buffers scale with it. Those are
small next to a prefill activation workspace, and they are zero-sized unless you
asked for embeddings.

So the memory is not a dormant prompt graph sitting there doing nothing. It is
one workspace, sized for the biggest thing that might run in it. During
single-stream decode most of it is idle, and lowering `-ub` is how you tell
llama.cpp not to reserve that much in the first place.

## Prefill and decode are not the same shape

The other half of the puzzle is that the two phases of inference have completely
different amounts of parallelism available.

<div style="margin: 1.5em 0;">
  <img src="/images/ubatch/prefill-vs-decode.svg"
       alt="Prefill drawn as a small number of fat chunks of 128 tokens each. Decode drawn as a long string of thin single-token steps."
       style="width: 100%; height: auto; display: block;"/>
</div>

During prefill the prompt tokens are already known, so many positions can go
into one ubatch. During generation each token has to be sampled before the next
one exists, so a single sequence contributes one row per ubatch. `-ub` is a
ceiling that prefill actually reaches and that single-stream decode almost never
does. That is the answer to the question I started with: the knob was never
about how many conversations you have, so having only one does not make it
irrelevant.

Worth noting a ubatch is not padded up to `n_ubatch`. The graph is built for
the actual token count, so a one-token decode really is a one-row matmul rather
than one real row and 127 zeros.

There *is* padding in the picture, but it is on the other axis. Attention reads
the whole prefix, so besides the rows going in, every graph has a second
dimension: `n_kv`, how many cached positions the attention step will look at.
That number naturally grows by one per generated token, and if the graph shapes
tracked it exactly, every single decode step would produce a differently shaped
graph that had to be rebuilt and reallocated from scratch.

So llama.cpp rounds it up. `llama_kv_cache::get_n_kv()` reports not the true
number of occupied cells but that number padded up to the next multiple of a
fixed quantum:

```cpp
// pad the n_kv value so that the graph remains constant across batches and can be reused
const uint32_t n_pad_cur = std::max(n_pad, 256u);
...
result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
```

With a quantum of 256, a cache holding 700 live positions is reported as 768,
and it keeps being reported as 768 until position 769 arrives. So 256
consecutive decode steps all build the identical graph, and only every 256th step
pays for a rebuild. The cost is that attention does a little arithmetic over
cells that are not really occupied, which the mask discards anyway.

As for where 256 itself comes from: it is a floor hardcoded in that line. The
model-specific `n_pad` passed in when the cache is constructed is 1 for the
paths I looked at in `llama-model.cpp`, so the `std::max` picks 256 and the model
value never matters. It is a tuning constant, not a property of any model, and
the comment above it gives two reasons: graph reuse, and that some backends
simply run faster on that alignment.

Note this is padding of a reported *count*, not extra allocation. The KV cache
itself was already allocated at its full `-c` size when the context was created.

## Part 1: putting different conversations in the same ubatch

Here is the part I found most interesting. When several conversations are active,
llama.cpp does not process them one after another, and it does not run separate
forward passes for each. It puts their tokens in the same ubatch.

<div style="margin: 1.5em 0;">
  <img src="/images/ubatch/unified-batch.svg"
       alt="A ubatch of six rows. Each row shows a token, a sequence id, and a position. Rows belong to three different conversations at unrelated positions, interleaved in arrival order rather than grouped."
       style="width: 100%; height: auto; display: block;"/>
</div>

This is continuous batching. Requests join and leave the running work as they
arrive and finish, rather than waiting for a group to be assembled and drained.
In the source this is `split_simple()`, which fills a ubatch by taking token
slots in submission order up to `n_ubatch`, and does not look at sequence ids at
all while doing it.

That is for the default unified KV cache. There is a second path, used when the
cache is split per sequence, where `split_equal()` builds ubatches from
equal-length chunks with ordered sequence ids. So "one ubatch mixes freely" is
true for the common case but not unconditionally true, which is the kind of
detail worth checking in the source rather than assuming.

The picture above raises an obvious question. Row 0 and row 3 are strangers.
They are about to be multiplied by the same weight matrices in the same kernel.
What stops one from leaking into the other?

### The mask is doing the work

Attention is the one place in the layer where rows read other rows, and the mask
is what decides which reads are allowed.

<div style="margin: 1.5em 0;">
  <img src="/images/ubatch/mask.svg"
       alt="A six by six grid of queries against keys. Cells are ticked only where the query and key belong to the same sequence and the key position is not in the future. Cells crossing sequences are greyed out, cells within a sequence pointing at future positions are marked in red."
       style="width: 100%; height: auto; display: block;"/>
</div>

The rule has two halves. A query can see a key only if they belong to the same
sequence, and only if the key is not in the future. The first half separates the
conversations, the second half is ordinary causal masking. Everything else
becomes negative infinity before the softmax, which means those keys contribute
nothing at all to the result.

The real thing in `set_input_kq_mask_impl` has more conditions than my two:

```cpp
if (cells.is_empty(j))            continue;   // unused cache slot
if (!cells.seq_has(j, seq_id))    continue;   // different sequence
if (causal && cells.pos_get(j) > p1) continue;   // future position
if (swa && is_masked_swa(...))    continue;   // outside the sliding window
```

There is an empty-slot check, a `causal_attn` flag that can turn causality off
entirely for models that want bidirectional attention, and sliding-window
masking for the models that use it. The two conditions I care about are in
there, alongside a few others.

One detail I liked: the sequence id used for masking is
`ubatch->seq_id[i][0]`, the first entry in the list. So the multi-sequence
membership I mentioned earlier is asymmetric, a cached cell can belong to
several sequences and be visible to all of them, but a query masks against one.

### Chunked prefill is the same mechanism

Once the mask is in view, chunking a long prompt stops looking like a separate
feature. The chunks are strictly ordered, chunk two cannot run before chunk one,
because chunk two's queries need chunk one's keys. But within a chunk the
positions go through together, and the mask sorts out their internal ordering.

<div style="margin: 1.5em 0;">
  <img src="/images/ubatch/chunked-prefill.svg"
       alt="Left: a full causal triangle shaded as one live computation. Right: the same triangle cut into horizontal strips, with one strip highlighted as live and the strips above it greyed out as already computed and stored in the KV cache."
       style="width: 100%; height: auto; display: block;"/>
</div>

A chunk of `U` new tokens against `C` already cached produces a `[U, C+U]` score
matrix rather than `[C, C]`. The area under the triangle is about the same
either way, so the total arithmetic does not change much. What changes is how
much of it is in flight at once, which is exactly the workspace the reservation
was sized for.

This is also why a smaller `-ub` cannot rescue you from a KV cache problem.
Later chunks still read the whole accumulated history. You have made the live
piece smaller, not the history.

## Part 2: why the weights do not care which conversation a row came from

The other half of the story is the feed-forward block, and it turns out to be
the simpler half once you stop thinking about it as a transformer thing.

In an ordinary neural network, the weights of a layer do not depend on which
sample a row came from. Ten samples means ten rows in and ten rows out, and the
same weight matrix multiplies all of them. Row three's output is built from row
three's input and nothing else.

The feed-forward block is exactly that, with tokens in the role of samples.

<div style="margin: 1.5em 0;">
  <img src="/images/ubatch/mlp-rows.svg"
       alt="Four input rows belonging to three different sequences, a single shared weight matrix W, and four output rows in matching colours. Each input row maps to its own output row, with no cross connections."
       style="width: 100%; height: auto; display: block;"/>
</div>

`[T, d] @ [d, f]` gives `[T, f]`, and output row `i` depends only on input row
`i`. It makes no difference whether the rows are consecutive positions from one
prompt or unrelated tokens from four different users. The weights are the same,
and the rows do not interact.

This is also where continuous batching earns its keep. Single-stream decode is
mostly waiting on memory: you read an entire weight matrix to process one row.
With a full ubatch you read the same weights once and amortise them across every
row in it. That is the throughput story, and it is a direct consequence of the
sharing.

### The output projection, the case worth thinking through

The case that most deserves a careful look is multi-head attention's output
projection. Each head produces its own output, they get concatenated, and `W_O`
mixes them. Mixing is the point of that layer. So what stops it from mixing
across tokens too?

<div style="margin: 1.5em 0;">
  <img src="/images/ubatch/output-projection.svg"
       alt="Panel A shows one token's head outputs concatenated along the feature axis and projected by W_O into a single output vector. Panel B shows the full matrix of rows from different sequences, with a crossed-out arrow from one row into a different row's output."
       style="width: 100%; height: auto; display: block;"/>
</div>

The answer is that the concatenation happens along the *feature* axis, not the
row axis. For a single token, its head outputs sit side by side in one row, and
`W_O` contracts along that direction. Head 3's contribution to token `i` gets
blended with head 7's contribution to token `i`. There is no arrangement of
numbers in `W_O` that would let it reach a different row, because the matmul
sums over features and carries the row index straight through.

So `W_O` is a row-wise operation in exactly the same sense the feed-forward
block is. It is a bigger, more interesting row-wise operation, but structurally
it is the same kind of thing.

And the rows going into it are already clean. Each head's output for token `i`
is a weighted sum of value vectors, with the weights coming from the masked
softmax. Anything from another conversation had weight zero. So it is clean rows
in, a row-wise operation, clean rows out.

## Walking the layer

Having convinced myself about `W_O`, I went through a whole layer to see whether
anything else crosses rows.

<div style="margin: 1.5em 0;">
  <img src="/images/ubatch/layer-audit.svg"
       alt="A transformer layer drawn as ten stacked steps. Nine are marked row-wise in green: input, RMSNorm, QKV projections, RoPE, output projection, residual adds, second RMSNorm, and the feed-forward block. One step, the masked softmax attention, is marked in red as crossing rows, with the mask attached to it."
       style="width: 100%; height: auto; display: block;"/>
</div>

RMSNorm normalises within a row, over that token's own features. The Q, K and V
projections are one matmul applied per row. RoPE uses each token's own position,
which is worth pausing on for a mixed ubatch: `inp_pos` is a tensor of length
`n_tokens` copied straight from `ubatch->pos`, so a row at position 4021 and a
row at position 17 sitting next to each other each get their own rotation. The
residual adds are elementwise. The feed-forward block is the shared-weights case
above.

That leaves one step that reads across rows, and it is the one wearing the mask.

This is a satisfying place to land, because it turns "is continuous batching
safe" from a question about the whole architecture into a question about one
operation. It also explains why the jumbled ordering in the ubatch
diagram costs nothing. Nothing in the layer cares what order the rows are in,
because nine of the ten steps never look at their neighbours and the tenth
consults the mask rather than the layout.

## Back to the VRAM question

With the vocabulary sorted out, the practical part is short:

- Lowering `-ub` shrinks the reserved workspace, and that is real, even at
  `-np 1`. The reservation is worst-case and driven by `min(n_ctx, n_ubatch)`.
- It does not touch the KV cache. That is `-c`, the KV precision, and how many
  contexts you provision.
- Lowering `-b` does not shrink the workspace. It caps the list length at the
  API boundary, and the workspace is sized from the ubatch split that happens
  after that.
- It costs prefill throughput and time to first token, not steady-state decode
  speed, since decode was already one row per ubatch.
- Queueing requests instead of provisioning slots for them is a reasonable
  trade. If you know you will never serve four users at once, `-np 1 -c 8192
  -ub 64` gives one request the full context and processes prompts in small
  pieces.

The current defaults, from `common/common.h`, are `n_batch = 2048`,
`n_ubatch = 512`, `n_parallel = 1`.

I have not done a careful sweep on my own hardware yet. When I do, the two
numbers I plan to watch are the `compute buffer size` line in the startup log
and prompt-processing tokens per second, across `-ub` values of 512, 256, 128
and 64. I would expect the buffer to shrink steadily and the prefill speed to
fall off somewhere below the point where the GPU stays busy, but I would rather
measure it than guess where that is.

## What I took away

The thing I keep coming back to is that the row dimension of a ubatch behaves
like a batch dimension in the ordinary sense, for every operation except one.
Attention is the exception, and attention has a mask. Continuous batching and
weight sharing are both consequences of that, rather than two separate tricks.

The naming is unfortunate. The knob everyone reaches for is spelled with batch
in it, the thing it controls is not the batch, and the thing that is called the
batch is mostly an API formality.

Two things I want to measure next: how much of the workspace reservation is
genuinely unavoidable versus conservative, and where the remaining prefill time
goes at small `-ub`. The terminology, at least, is now settled.

Source references in this post are against llama.cpp master as of August 2026,
mainly `include/llama.h`, `src/llama-context.cpp`, `src/llama-kv-cache.cpp` and
`src/llama-graph.cpp`. If something here is wrong I would like to know.
