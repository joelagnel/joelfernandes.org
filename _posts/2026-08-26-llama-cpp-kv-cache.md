---
layout: post
title: "The KV cache in llama.cpp"
date: 2026-08-26
categories: [machine-learning, inference]
tags: [machine-learning, ml, llama-cpp, inference, transformers, attention, kv-cache, vram, gqa, quantization]
author: Joel Fernandes
description: "Building up what a KV cache cell is from what attention actually needs, then following it through llama.cpp: the metadata columns, slot allocation, the server's prefix reuse, and measurements of how the pre-allocated cache scales with context size and -np on an RTX 4090."
published: false
---

The KV cache is the largest allocation in an inference server after the model
weights, and it is allocated once, at startup, at full size. Everything about
its cost is decided before the first token arrives.

I wanted to understand what is actually in it. Not the formula for its size,
which is easy to look up, but the shape of the thing: what one token
contributes, how the code knows which stored vector belongs to which
conversation, and what happens when the space runs out. This post builds that
up from what attention needs, then follows it into llama.cpp. The numbers at
the end are measured on an RTX 4090 with Qwen3-4B.

Code references are to llama.cpp at commit `4d19b28`, and the measurements use
release build `b10635`. Line numbers drift, so function names are the anchors.

## The problem the cache solves

Attention works by comparison. To produce the next token, the model takes the
current token's **query** vector and compares it against a **key** vector
belonging to every earlier token in the sequence. Those comparisons become
weights, and the weights mix together a **value** vector from each of those same
earlier tokens. Keys and values, hence KV.

The property that makes caching possible is that a token's key and value depend
only on that token and its position. They do not depend on anything that comes
after it. Take the sentence "the cat sat on the". When the model generates the
sixth token, it compares against keys for "the", "cat", "sat", "on", "the". When
it then generates a seventh, it compares against those same five keys plus one
new one. The first five are bit-for-bit identical to what they were a step
earlier.

So there are two options. Recompute all of them on every step, which makes
generating an *n*-token reply cost work proportional to *n²*. Or compute each
one once and keep it, which makes it proportional to *n*. The store that keeps
them is the KV cache.

That framing already tells you the cache is write-once. A stored key is never
updated, only written, read many times, and eventually discarded. This is worth
holding onto, because it explains why the data structure has no update path and
why "eviction" turns out to mean something narrower than it does in a CPU cache.

## Two things called the cache

The word covers two separate mechanisms in llama.cpp, and they answer different
questions. For the rest of this post:

- **The KV cache** means the library-level tensor store in
  `src/llama-kv-cache.cpp`, holding the K and V vectors themselves, allocated
  once at context creation. This is the thing that consumes VRAM.
- **The prompt cache** means the server-level machinery in `tools/server/`,
  which remembers *which tokens* a slot already processed, so their K and V do
  not have to be recomputed. This is bookkeeping layered over the first one.

Questions about eviction and prefix lookup are questions about the second.
Questions about memory are questions about the first. I start with the first.

## What one token leaves behind

Start with a single token and follow it. The model processes `"sat"` at
position 7 of a conversation. Qwen3-4B has 36 transformer layers, and attention
happens independently in each one, so the token produces a key and a value in
every one of them: 36 keys and 36 values.

Each of those vectors has a width. For keys it is `n_embd_k_gqa`, which
`src/llama-hparams.cpp` computes as `n_embd_head_k(il) * n_head_kv(il)`, the
width of one attention head multiplied by the number of key/value heads. Qwen3-4B
has 32 attention heads but only 8 key/value heads, with a head width of 128, so
the number is 8 × 128 = 1024 values per key row, and the same for values.

That 8-versus-32 is grouped-query attention. Several query heads share one
key/value head, so the cache stores 8 rows' worth of keys instead of 32. Under
plain multi-head attention the row would be 32 × 128 = 4096 values. The cache is
a quarter of the size it would otherwise be, and this is a property baked into
the model, not a flag you set.

<div style="margin: 1.5em 0;">
  <img src="/images/kvcache/one-token.svg"
       alt="A token labelled sat, position 7, chat A, feeds a stack of transformer layers 0, 1, 2 through 35. Each layer contains a K row and a V row labelled row 7. A purple bracket spans all the layers and is labelled row index 7 in all 72 tensors, that one shared index is what the code calls a cell. A strip at the bottom gives the arithmetic: 1024 values per row times 2 bytes times 2 rows times 36 layers equals 147456 bytes equals 144 KiB for this one token."
       style="width: 100%; height: auto; display: block;"/>
</div>

So one token at `f16` costs:

```
1024 values × 2 bytes × 2 (K and V) × 36 layers = 147,456 bytes = 144 KiB
```

Under multi-head attention the same token would cost 576 KiB. Every sizing
number later in this post is that 144 KiB multiplied by a count of tokens.

## Why a single index is enough

Now the storage question. Where do those 72 vectors go?

They go into 72 pre-allocated tensors, two per layer, created in the
`llama_kv_cache` constructor and named `cache_k_l0`, `cache_v_l0`, `cache_k_l1`
and so on:

```cpp
ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;
```

Read the shape as: each tensor is `kv_size` rows tall, and each row is one
token's vector for that layer. `kv_size` is the total number of tokens the cache
can hold, fixed at startup.

The convenient part is that the layers do not need independent bookkeeping. When
the token at position 7 is written, it is written to row 7 of `cache_k_l0`, row
7 of `cache_v_l0`, row 7 of `cache_k_l1`, and so on through all 72 tensors. The
same row index everywhere. This is visible in the code: `cpy_k` receives a
single index tensor `k_idxs` and applies it per layer, and that tensor is built
once per batch by `set_input_k_idxs`, not once per layer.

That shared row index is the unit the codebase reasons about, and it calls it a
**cell**. A cell is not a struct. There is no `llama_kv_cell` type to go read.
Cell 7 is the statement "row 7, in every tensor, belongs to one token", plus
some facts about which token that is.

Saying "the cache has 4096 cells" is therefore the same as saying every one of
its 72 tensors is 4096 rows tall, and it can hold 4096 tokens in total.

## What the cache has to remember besides numbers

A row of 1024 floating point numbers, on its own, is not usable. To decide
whether to include a stored key in a comparison, the code has to answer
questions about it, and the raw vector answers none of them.

Work out what those questions are, and the metadata falls out.

**Which conversation is this key from?** A server handling four chats at once
puts all of their keys in the same tensors. A query from chat A must never be
compared against a key from chat B, or the two conversations bleed into each
other. So each cell needs to record its owner.

Except "owner", singular, is too weak. Two chats that start with the same system
prompt produce byte-identical keys for those tokens, because a key depends only
on the token and its position. Storing them twice would be pure waste. The
useful thing is to store them once and let both chats read them. That requires
a cell to record a *set* of owners, not one.

In llama.cpp that set is `seq[i]`, a `std::bitset<LLAMA_MAX_SEQ>` where
`LLAMA_MAX_SEQ` is 256, defined in `src/llama-cparams.h`. It is a bitmap over
sequence ids: bit A set means chat A can read this cell. Prefix sharing between
conversations is then free at the storage level, because it is one extra bit.

**Where in its sequence does this key sit?** Two reasons this is needed. First,
a token may not attend to its own future. When the query is at position 5, keys
at positions 6 and 7 have to be excluded, and comparing indices does not work
because cells are not filled in order. Second, sliding-window models only attend
to the last *n* positions, which is also a statement about position.

That field is `pos[i]`, an `int32`. It doubles as the free marker: `pos[i] == -1`
means the cell holds nothing, which is exactly what `is_empty()` tests.

**Has this key's position been renumbered since it was written?** This one is
less obvious. Positions are not just labels. They are baked into the key vector
itself by rotary position embedding, which rotates the vector by an angle
derived from the position. Change a token's position and its stored key becomes
wrong, and fixing it means re-running that rotation on the GPU.

Sometimes positions do change: the server drops a span from the middle of a full
context and everything after it shifts down. Doing a GPU pass per affected cell
at the moment of the change would be wasteful, so the code writes down how far
each cell has moved and applies the whole batch of rotations later, on the next
`update()`. That pending delta is `shift[i]`.

**Anything else?** One field, `ext[i]`, carries 2D spatial coordinates for
multimodal RoPE, where image patches have an x and a y rather than a single
sequence position. It is inert for text-only models.

So four columns, one per question. Together with the tensor rows they index,
that is the whole cell:

<div style="margin: 1.5em 0;">
  <img src="/images/kvcache/cell-and-tensors.svg"
       alt="A table with columns cell, pos, seq, shift, K row and V row, and eight rows. Cells 0 to 2 have positions 0, 1 and 2 and a sequence set containing both A and B, shaded to show the K and V rows are shared. Cells 3 and 4 belong to chat A at positions 3 and 4. Cell 6 belongs to chat B at position 3. Cells 5 and 7 have pos of -1, an empty sequence set, and their K and V rows drawn dashed and marked unused. Annotations note that cells 0 to 2 carry two bits so the system prompt is stored once and read by both chats, and that pos equals -1 means free while the vectors are still physically there but nothing reads them."
       style="width: 100%; height: auto; display: block;"/>
</div>

The diagram shows a small server state: chat A and chat B were opened with the
same three-token system prompt, A is a few turns in, B has just started. Cells
0, 1 and 2 carry two bits each. Their key and value rows exist once and both
chats read them.

In the source, those four columns are four separate `std::vector`s inside
`llama_kv_cells` in `src/llama-kv-cells.h`, indexed in parallel:

```cpp
    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    std::set<uint32_t> used;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    std::vector<llama_pos> shift;

    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;
```

Parallel arrays rather than one array of structs. The fifth member, `used`, is a
sorted set of the occupied indices, and it exists to answer "what is the highest
live cell" in constant time. That question turns out to drive performance, and I
come back to it.

## Where those fields get read

The metadata pays for itself in exactly one place, and seeing that place is what
made the design click for me. Before attention runs, the code builds a **mask**:
a grid with one row per query token and one column per cell, holding 0 where the
comparison should count and negative infinity where it should be thrown away.
Adding negative infinity before the softmax drives that weight to zero.

Building one row of that grid is a loop over cells, and its body is three tests:

<div style="margin: 1.5em 0;">
  <img src="/images/kvcache/mask-row.svg"
       alt="A query token from chat A at position 5 sits at the left. To the right, nine cells are shown with their pos and seq values: cells 0 to 5 belong to chat A at positions 0 to 5, cell 6 belongs to chat B at position 3, cell 7 belongs to chat A at position 6, and cell 8 is empty with pos -1. Beneath each cell is its mask value: 0 and read it for cells 0 to 5, negative infinity and other chat for cell 6, negative infinity and not yet for cell 7, negative infinity and nothing here for cell 8. A panel below lists the three tests the loop runs in order: is_empty, seq_has, and pos_get greater than p1, with the reason for each."
       style="width: 100%; height: auto; display: block;"/>
</div>

The code in `set_input_kq_mask_impl` reads as the diagram does:

```cpp
                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }
```

where `skip` writes `-INFINITY` into the mask and `p1` is the query's own
position. `seq` answers the second test and `pos` answers the third. Remove
either column and the row cannot be built.

Two things follow from this that matter later.

The loop runs from `j = 0` to `j = n_kv`, so the width of that mask is the width
of the attention work, and it is set by how far the cell array is occupied, not
by how many tokens the current query actually needs. Cells belonging to other
sequences are visited, tested, and written off as negative infinity. The
comparison hardware still processes that column.

And the write is one element per cell per query token, on the CPU, which is why
recent versions added a shortcut: tokens of the same sequence in one batch share
almost all of their mask row, so the code copies the previous row and only
revisits the cells whose verdict could differ.

## The life of a cell

With the fields established, the state machine is small.

<div style="margin: 1.5em 0;">
  <img src="/images/kvcache/cell-life.svg"
       alt="Four boxes in a row. Free, with pos -1 and an empty sequence set. An arrow labelled find_slot and pos_set leads to Written, with pos 12 and sequence set containing A. An arrow labelled seq_cp A to B and one more bit leads to Shared, with pos 12 and sequence set containing A and B. An arrow labelled seq_rm B then seq_rm A leads to Free again. A panel below explains that dropping chat B does not free the cell, that seq[i].reset(B) clears one bit and the cell empties only when seq[i].none() is true, and that there is no reference count, timestamp or eviction score. A closing note says the second route back to free is the sliding-window test inside find_slot."
       style="width: 100%; height: auto; display: block;"/>
</div>

A cell starts free. `find_slot()` picks it, `pos_set()` stamps a position on it,
and a sequence bit goes in. If a second conversation is branched from the first,
`seq_cp` sets another bit and copies no vector data at all. Dropping a
conversation clears one bit, and only when the last bit goes does the cell
become free:

```cpp
        seq[i].reset(seq_id);
        seq_pos_dec(seq_id, pos[i]);

        if (seq[i].none()) {
            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;
            used.erase(i);
            return true;
        }
        return false;
```

That is the entire reclamation rule inside the library. The bitset is doing the
job a reference count would do, one bit per possible owner.

Note what freeing does not do: it does not touch the tensor rows. The 1024
floating point numbers are still sitting in VRAM. Setting `pos[i] = -1` makes
the mask loop skip the column, which makes the data unreachable, which is all
that "free" needs to mean here.

## Finding a free cell

There is no hash table and no content-addressed lookup anywhere in the library.
Nothing hashes a token to find where its key lives. `find_slot()` in
`src/llama-kv-cache.cpp` walks the cell array from a cursor and takes the first
cells that are available.

Each stream keeps a `head` cursor, described in the header as "the current index
from where we start searching for a free slot in the ring buffer of KV cells".
The scan begins there, with one adjustment:

```cpp
        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }
```

The per-cell test is short, and it contains the only automatic reclamation the
library performs:

```cpp
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    ...
                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                }
```

Read that as two conditions. The cell is empty, or it holds exactly one sequence
and that sequence's sliding window has already moved past it. Sliding window
attention means a layer only ever attends to the last `n_swa` positions, so a
key older than the window can never be read again, and its cell is dead whether
or not anyone released it. For a model without sliding windows, `is_masked_swa`
returns false for the `LLAMA_SWA_TYPE_NONE` case and this branch never fires.

<div style="margin: 1.5em 0;">
  <img src="/images/kvcache/find-slot.svg"
       alt="Two rows of fourteen cells. In the before row, cells 0 to 2 hold chat A and cells 3 and 4 hold chat B, cell 9 holds a chat B entry shown in orange and annotated as outside its sliding window, and the rest are dashed and empty, with a red arrow marking the head cursor at cell 5. In the after row, cells 5, 6 and 7 are filled in green with chat A. A caption notes that cell 9 was equally eligible and the scan simply reached cell 5 first, and that the three tokens need not be adjacent, which is why writes go through an index tensor rather than a memcpy at the cursor."
       style="width: 100%; height: auto; display: block;"/>
</div>

Placement does not have to be contiguous. `prepare()` calls
`find_slot(ubatch, false)`, and that flag is what decides:

```cpp
    const uint32_t n_test = cont ? n_tokens : 1;
```

With `cont` false, one batch's tokens can land in scattered cells. That is why
writes go through `ggml_set_rows` with an index tensor rather than a `memcpy` at
the cursor: the destination rows are not adjacent.

That choice removed a feature. Older versions defragmented the cache to close
holes, and current master does not:

```c
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)
```

The `-dt` flag still parses and does nothing, logging `DEPRECATED:
--defrag-thold is deprecated and no longer necessary to specify`. It is
unnecessary because scattered placement makes holes directly usable. It is not
entirely free, for a reason that arrives in the section on `n_kv`.

## The library has no eviction policy

Searching the library sources for `lru` or `least recently` returns nothing.
There is no recency tracking, no importance score, and no automatic reclamation
of individual cells beyond the sliding-window test above.

What exists instead is an explicit API that the caller drives:

| Call | Effect |
|---|---|
| `clear(data)` | Reset every cell in every stream, optionally zero the buffer |
| `seq_rm(seq, p0, p1)` | Drop a sequence from cells whose position is in `[p0, p1)` |
| `seq_cp(src, dst, p0, p1)` | Add `dst` to the same cells, no data copy within a stream |
| `seq_keep(seq)` | Empty every cell that does not contain `seq` |
| `seq_add(seq, p0, p1, delta)` | Renumber positions, deferring the RoPE fixup |
| `seq_div(seq, p0, p1, d)` | Divide positions, same deferral |

`seq_add` is the deferral described earlier: it sets a `has_shift` flag and
accumulates into `shift[i]`, and the next `update()` builds and runs a RoPE
graph over the affected keys in one go.

So when memory pressure needs a policy, the policy lives a level up.

## What the server does instead

The server owns `n_parallel` slots, each with its own token list and its own
share of the cache. `get_available_slot()` picks one in three stages.

First, an explicit `id_slot` if the request named one. Second, longest common
prefix similarity: for each idle slot, compute how much of the incoming prompt
that slot already holds.

```cpp
                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();
```

The bar is `--slot-prompt-similarity`, default 0.1. Third, only if no slot
cleared that bar, plain least-recently-used:

```cpp
        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;
            for (server_slot & slot : slots) {
                if (slot.is_processing()) { continue; }
                if (!ret || slot.t_last_used <= t_last) {
```

Prefix match first, recency only as a fallback. Both operate on whole slots,
never on individual cells.

When a slot's own context fills, the server can shift it, dropping a span from
the middle and renumbering what follows:

```cpp
                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
```

`n_keep` protects a prefix and `n_discard` defaults to half of what remains.
This is where `shift[i]` earns its place: a `seq_add` over tens of thousands of
cells becomes one queued RoPE pass rather than tens of thousands of small ones.

Context shift is off by default (`bool ctx_shift = false;`). With it off,
generation stops cleanly when the slot is full, setting `STOP_TYPE_LIMIT`, and a
prompt larger than the slot is rejected up front with
`ERROR_TYPE_EXCEED_CONTEXT_SIZE` rather than silently truncated.

## How prefix reuse works

The core is one line. When a request arrives at a slot, the server compares the
slot's existing token list against the new prompt:

```cpp
                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);
```

`get_common_prefix` is a token-by-token walk that stops at the first mismatch.
Everything before `n_past` is already in cells with correct positions, so only
the suffix needs a forward pass. Appending a turn to a conversation reuses
everything; editing the first sentence of a long prompt reuses nothing after it,
because every later token's position has changed and therefore every later
token's key is wrong.

One wrinkle shows up in server logs:

```cpp
                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
```

If the new prompt is exactly a prefix of what the slot holds, one token is
re-evaluated anyway, because the forward pass is what produces the logits needed
to sample the next token. The cache stores keys and values; it does not store
the output distribution.

`--cache-reuse` relaxes the strict-prefix constraint. Past the common prefix it
walks two cursors looking for matching runs, and slides any run longer than the
threshold into its new position:

```cpp
                                        if (n_match >= (size_t) n_cache_reuse) {
                                            ...
                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);
```

The stale span is removed and the matching span is renumbered by `kv_shift`,
which is the deferred-RoPE path again. It requires `llama_memory_can_shift`, is
disabled for multimodal prompts, and is off by default; several bundled presets
set it to 256.

Beyond a slot's own history, the server keeps a RAM prompt cache, sized by
`--cache-ram` at a default of 8192 MiB. Each entry holds the token list, the
serialized per-sequence KV state from `llama_state_seq_get_data_ext`, the draft
model's state if speculative decoding is on, and any context checkpoints.
Selection balances two ratios:

```cpp
        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();
        ...
        // don't trash large prompts
        if (f_keep_cur < 0.25f) { continue; }
```

so an entry is only reused if the new prompt matches a lot of *it*, not merely
if it matches a lot of the new prompt. A 40k-token cached conversation is not
evicted to serve a 200-token request that happens to share its system prompt.
Loading moves the entry out of the cache rather than copying it. This cache is
RAM only; `--slot-save-path` is the separate, explicit route to disk.

## How the size is determined

The cache is allocated once, in the `llama_kv_cache` constructor, via
`ggml_backend_alloc_ctx_tensors_from_buft`. It never grows and never shrinks.
From the tensor shapes, the total is:

```
bytes = Σ over cached layers  row_size(type_k, n_embd_k_gqa) * kv_size * n_stream
      + Σ over cached layers  row_size(type_v, n_embd_v_gqa) * kv_size * n_stream
```

where `row_size` for `f16` is `2n`, for `q8_0` is `34n/32`, and for `q4_0` is
`18n/32`, from `ggml_row_size` and the block sizes of those types. This is the
per-token 144 KiB from earlier, multiplied by the cell count.

`kv_size` is `n_ctx_seq`, and how that relates to `--ctx-size` depends on the
cache mode:

```cpp
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);
```

With a split cache, `-c 8192 -np 4` gives each sequence 2048 tokens, not 8192.
Total allocation is the same either way.

### Measured

Qwen3-4B has 36 layers, 8 KV heads and `head_dim` 128, so `n_embd_k_gqa` and
`n_embd_v_gqa` are both 1024, and one token costs 144 KiB at `f16` as computed
above. Sweeping `llama-server` on an RTX 4090 and reading the
`llama_kv_cache: size` line it prints at startup:

| flags | cells | reported cache | formula | GPU delta |
|---|---|---|---|---|
| `-c 1024 -np 1` | 1024 | 144.00 MiB | 144.00 | 2632 MiB |
| `-c 4096 -np 1` | 4096 | 576.00 MiB | 576.00 | 3067 MiB |
| `-c 8192 -np 1` | 8192 | 1152.00 MiB | 1152.00 | 3647 MiB |
| `-c 16384 -np 1` | 16384 | 2304.00 MiB | 2304.00 | 4807 MiB |
| `-c 32768 -np 1` | 32768 | 4608.00 MiB | 4608.00 | 7127 MiB |
| `-c 40960 -np 1` | 40960 | 5760.00 MiB | 5760.00 | 8287 MiB |

Linear in the cell count, and exact to the byte in all six cases. The 40960 row
is the model's full trained context and costs 5.6 GiB of cache on top of 2.3 GiB
of weights.

Now `-np`, holding `-c` fixed at 16384:

| flags | cells per stream | streams | reported cache |
|---|---|---|---|
| `-np 1` | 16384 | 1 | 2304.00 MiB |
| `-np 2` | 8192 | 2 | 2304.00 MiB |
| `-np 4` | 4096 | 4 | 2304.00 MiB |
| `-np 8` | 2048 | 8 | 2304.00 MiB |
| `-np 16` | 1024 | 16 | 2304.00 MiB |

**`-np` does not cost memory.** It divides the memory already requested. Sixteen
slots of 1024 tokens costs exactly what one slot of 16384 costs. What `-np` buys
is concurrency; what it costs is per-conversation context length. Asking for
`-c 16384 -np 16` and expecting each client to get 16k is the mistake, and the
server says so at startup: `n_ctx_seq (1024) < n_ctx_train (40960)`.

## Unified and non-unified

`kv_unified` is not a separate class. It is the stream count:

```cpp
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
```

One shared cell array for all sequences, or one array per sequence. The library
default is `false`, but the stock server invocation turns it on, because `-np`
defaults to `-1` meaning auto:

```cpp
        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
```

Unified has real advantages, and they follow from the `seq` bitset. Sequences
can share cells, so a common prompt prefix lives in memory once with several
bits set. Copying a sequence is metadata only. And any one sequence can use the
entire context rather than a fixed slice.

The cost lands on that mask loop. `get_n_kv()` decides how wide the attention
graph is built:

```cpp
    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }
```

`used_max_p1()` is `*used.rbegin() + 1`, the highest occupied index in the
array. This is the `used` set from earlier earning its keep. Note it is a
property of the array, not of any one sequence. With one shared array, whichever
sequence has pushed furthest sets `n_kv` for every sequence in it.

That 256 is a literal in this function, and the two comments above it give both
reasons: a constant graph shape lets the graph be reused across batches, and
some backends run the kernels faster at that alignment. `GGML_PAD` is
`(((x) + (n) - 1) & ~((n) - 1))`, a power-of-two round-up, which is why the
constant is 256 rather than, say, 200. The `n_pad` member is passed as `1` from
every construction site in `llama-model.cpp`, so in practice the padding is
always 256.

`n_kv` is then literally the width of the mask tensor, and in unified mode there
is one mask plane covering all sequences:

```cpp
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
    ...
    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
```

and this is the third test from the mask diagram, seen from the cost side: a
cell belonging to another sequence gets `-INFINITY` written into the column, and
the attention kernel still walks that column.

Fragmentation compounds it. Cells freed by `seq_rm` can be anywhere, but
`used_max_p1()` only falls when the *topmost* live cell is freed. Ten live cells
at indices 0 to 9 plus one live cell at index 8000 still yields `n_kv` around
8192. With defrag removed, nothing compacts that. The heuristic in `find_slot`
that resets `head_cur` to 0 when there is plenty of free space below it is the
counter-pressure, and it is a heuristic.

<div style="margin: 1.5em 0;">
  <img src="/images/kvcache/unified-vs-split.svg"
       alt="Top: a single wide bar of 40960 cells with slot 0 occupying 31111 of them in blue and a small green block for slot 1 just after, with a red bracket spanning the occupied region labelled n_kv equals 31232 for every slot including slot 1. Bottom: four separate bars of 40960 cells each, one per stream, with stream 0 filled to 31111 in blue and stream 1 holding a sliver of green, and separate brackets showing n_kv equals 31232 for stream 0 and n_kv equals 256 for stream 1. A results box at the bottom gives the measured throughput numbers."
       style="width: 100%; height: auto; display: block;"/>
</div>

### Measuring it

The prediction is specific: in a unified cache, a short sequence should decode
more slowly when a long sequence is resident, even though its own content has
not changed. In a split cache it should not care.

The measurement is a within-configuration A/B, so model, quantization, batch
settings and hardware are constant across each comparison. Phase A decodes a
short request on slot 1 with the server otherwise empty. Phase B parks a
31,111-token prompt on slot 0 and decodes the identical short request again.
Both configurations give every slot a capacity of 40960 tokens, with `q4_0` KV
to keep the split allocation inside 24 GB.

The first two attempts showed no difference at all, and the reason was in the
server log:

```
srv init: idle slots will be saved to prompt cache and cleared upon starting a new task
```

`try_clear_idle_slots()` returns immediately unless the cache is unified:

```cpp
    bool try_clear_idle_slots() {
        bool res = false;
        if (!params_base.kv_unified) {
            return res;
        }
```

It exists precisely because of this problem. An idle slot's stale tokens hold
`used_max_p1()` high and inflate `n_kv` for every other sequence, so the server
purges them. That defence was evicting the long prompt before the short request
ran. Disabling it with `--no-cache-idle-slots` exposes the underlying behaviour:

| configuration | slot 1 alone | slot 1 with 31k parked | change |
|---|---|---|---|
| unified | 228.01 tok/s | 201.89 tok/s | 11.5% slower |
| non-unified | 235.10 tok/s | 231.27 tok/s | 1.6% |

Seven repetitions each, medians reported. The 1.6% on the split cache is within
run-to-run variation; the 11.5% is not.

So the mechanism is real and measurable, and the stock configuration already
mitigates it. Unified costs throughput when sequences of very different lengths
coexist and either the idle-slot purge is disabled or the long sequence is
genuinely active. The upstream header states the condition:

```cpp
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
```

Note the clause at the end. When the sequences *do* share a large prefix,
unified wins, because the shared cells exist once instead of `n_seq_max` times.

## Making cells cheaper

Four mechanisms reduce the footprint, and they are different in kind.

**GQA** is the largest and is not a setting. Qwen3-4B's 8 KV heads against 32
query heads make each row 1024 values instead of 4096, computed at the top.

**Cache quantization** via `-ctk` and `-ctv` shrinks the row itself. Nine types
are whitelisted in `common/arg.cpp`: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl,
q5_0, q5_1`. Measured on Qwen3-4B at `-c 32768`:

| `-ctk` / `-ctv` | reported cache | vs f16 | GPU delta |
|---|---|---|---|
| `f16` / `f16` | 4608.00 MiB | baseline | 7127 MiB |
| `q8_0` / `f16` | 3528.00 MiB | 0.77x | 6038 MiB |
| `q8_0` / `q8_0` | 2448.00 MiB | 0.53x | 4968 MiB |
| `q4_0` / `q4_0` | 1296.00 MiB | 0.28x | 3816 MiB |

Quantizing V requires flash attention, enforced in two places. Once in parameter
validation, where `auto` is silently upgraded:

```cpp
    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
```

and again as a hard backstop after autodetection:

```cpp
        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
```

The reason is a layout change. With flash attention off, the V tensor is stored
transposed, selected by `v_trans`:

```cpp
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();
```

The same bytes read with different strides, so the non-flash `KQV` matmul can
consume V directly. Transposed, one token's V is one *element* in each of 1024
rows rather than one contiguous row, which has two consequences. The index
tensor grows from `n_tokens` entries to `n_tokens * n_embd_v_gqa` entries, every
layer's V rows get padded to the widest layer in the model, and a single element
cannot be addressed inside a quantization block. Hence the requirement. There is
also a divisibility check: the head dimension must be a multiple of the type's
block size.

Current master additionally applies an orthonormal Walsh-Hadamard rotation
before quantizing, spreading outlier values across the block so a single large
value does not dominate the block's scale factor and coarsen every other value
in it:

```cpp
// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
```

It is decided separately for K and V, each gated on that side's cache type being
quantized and its head width being a multiple of 64:

```cpp
        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;
```

The matrices are generated once at construction, for every power-of-two width
from 64 up to the widest head in the model, and kept in host memory.
`LLAMA_ATTN_ROT_DISABLE` is the escape hatch.

**Sliding window attention** reduces the cell count rather than the row width.
`llama_kv_cache_iswa` holds two caches with complementary layer filters:

```cpp
    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/llama.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
```

so windowed layers get `n_swa + n_ubatch` cells rather than `n_ctx_seq`, which
is the same fact the `find_slot` reuse test exploits, applied at allocation time
instead of at placement time. Qwen3-4B does not use this; its config reports
`sliding_window: null`. `--swa-full` disables the saving, and exists because a
truncated window makes arbitrary prefix restore impossible.

**MLA** is genuine latent compression. For DeepSeek-family models, `has_v =
!is_mla` eliminates the V tensor entirely, so those cells have half as many
tensor rows behind them, and the stored K row is a `kv_lora_rank`-sized latent
expanded at attention time.

What is *not* implemented is worth stating, because these techniques get
discussed as though they were available. Searching the sources for
`streamingllm`, `snapkv`, `h2o`, `heavy.hitter` or `token.drop` returns nothing.
There is no importance-based token dropping and no heavy-hitter selection, which
is consistent with the metadata: no column records how often or how strongly a
cell has been attended to, so there is nothing to rank cells by. The `attn_sinks`
tensor that does exist is a learned per-head weight used by gpt-oss-style models
inside the softmax; it is not StreamingLLM's keep-the-first-N policy and it does
not manage the cache. The only policy that discards tokens is the server's
context shift, and it drops a contiguous span because of where it sits, not
because of what it contains.

## What I would tell someone sizing a server

Work out the per-token cost for your model the way it was worked out above:
`n_embd_k_gqa` plus `n_embd_v_gqa`, times the bytes per value, times the layer
count. For Qwen3-4B at `f16` that is 144 KiB. Multiply by `-c` and that is the
cache, exactly, before the process starts. `-np` splits that number rather than
multiplying it. `-ctk q8_0 -ctv q8_0 -fa on` roughly halves it, for a
quantization error worth measuring on your own workload. And if you run many
short concurrent conversations that do not share prefixes, `--no-kv-unified` is
worth an experiment, because on this hardware it recovered 11.5% of decode
throughput in the case it is designed for.

Two things I want to measure next: the perplexity cost of `q4_0` KV on a real
task rather than a synthetic one, and whether the fragmentation-driven `n_kv`
inflation is observable in a long-running server with churning conversations,
now that there is no defrag pass to clean up after it.

Measurements: llama.cpp `b10635`, Vulkan backend, RTX 4090 24564 MiB, driver
580.159.03, Qwen3-4B-Q4_K_M from `ggml-org/Qwen3-4B-GGUF`. Code quoted from
`src/llama-kv-cells.h`, `src/llama-kv-cache.cpp`, `src/llama-kv-cache-iswa.cpp`,
`src/llama-context.cpp`, `src/llama-graph.cpp`, `src/llama-hparams.cpp`,
`common/arg.cpp`, `common/common.h`, `tools/server/server.cpp`,
`tools/server/server-context.cpp` and `tools/server/server-task.{h,cpp}` at
commit `4d19b28`. If something here is wrong I would like to know.
