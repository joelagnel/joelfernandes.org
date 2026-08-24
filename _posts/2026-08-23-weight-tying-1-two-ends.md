---
layout: post
title: "Weight tying, part 1: the two ends of a transformer"
date: 2026-08-23
categories: [machine-learning, neural-networks, transformers]
tags: [machine-learning, ml, deep-learning, neural-networks, transformers, gpt-2, llm-c, embeddings, weight-tying, pytorch]
author: Joel Fernandes
description: "wte turns an index into a vector, lm_head turns a vector into 50257 scores. Working through both with real GPT-2 tensors, and why the two matrices end up the same shape."
published: true
---

I was reading `train_gpt2.py` in Andrej Karpathy's
[llm.c](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py) and stopped
at this line:

```python
self.transformer.wte.weight = self.lm_head.weight
```

The table the model reads at the input is set to the same object as the matrix
it multiplies by at the output. No transpose, no copy.

This is called weight tying, and it is a choice rather than a rule. GPT-2 ties.
Llama 3.2 1B and 3B tie, Llama 3.1 8B does not. Qwen3 ties up to 4B and stops at
8B. Two different labs, the same cutoff, which is a hint that something
size-dependent is going on.

That raises three separate questions, and they take three posts to answer:

1. **This post.** What the two matrices actually do, and why they have the same
   shape, which is what makes the assignment legal in the first place.
2. **[How one matrix means two different things](/blog/2026-08-23/weight-tying-2-one-matrix-two-meanings.html).** Why the same numbers
   can serve a lookup at the input and a prediction at the output, when those
   are not the same job.
3. **[What weight tying costs](/blog/2026-08-23/weight-tying-3-what-it-costs.html).** What sharing does to the gradients, and
   why large models stop doing it.

Every tensor below is printed from the real GPT-2 124M weights, not made up.

## Text becomes idx, a tensor of integers

Text does not reach the model as text. A tokenizer splits it into pieces and
looks each piece up in a fixed vocabulary of 50257 entries, so what arrives is
a list of integers.

```python
>>> import tiktoken
>>> enc = tiktoken.get_encoding("gpt2")
>>> enc.encode("The sky is")
[464, 6766, 318]
```

Two sequences stacked together make `idx`:

```python
>>> idx = torch.tensor([enc.encode("The sky is"),
...                     enc.encode("I ate an")])
>>> idx.shape
torch.Size([2, 3])
>>> idx
tensor([[  464,  6766,   318],
        [   40, 15063,   281]])
```

`idx` has shape `(b, t)`. `b` is the batch size, here 2, the number of
independent sequences going through together. `t` is the time dimension, here 3,
the token positions within one sequence. So `idx[0]` is the first sequence and
`idx[0][0]` is the single integer 464, which decodes back to `"The"`.

Everything in `idx` is one integer. Nothing is a vector yet. The number 464
carries no meaning on its own; it is a row number.
## wte, the table that turns an index into a vector

`wte` stands for word token embedding. It is the first thing the forward pass
touches:

```python
wte = nn.Embedding(config.vocab_size, config.n_embd)   # 50257 x 768
```

```python
>>> wte.weight.shape
torch.Size([50257, 768])
```

One row per vocabulary entry, 768 numbers per row. Row `i` is the model's
representation of token `i`, and those numbers are learned like any other
weight.

<div style="margin: 1.5em 0;">
  <img src="/images/wte-lmhead/wte-grid.svg"
       alt="A grid representing wte.weight with 50257 rows and 768 columns, row 4171 highlighted. An arrow pulls that row out to the right, shown as cells of floating point values labelled wte.weight[4171], the row for the token blue, 768 floats. Annotation reads 50257 x 768 = 38,597,376 parameters, 154.4 MB in fp32."
       style="width: 100%; height: auto; display: block;"/>
</div>

Calling `wte(idx)` replaces each integer with its row:

```python
>>> tok_emb = model.transformer.wte(idx)
>>> tok_emb.shape
torch.Size([2, 3, 768])
```

`(2, 3)` went in and `(2, 3, 768)` came out. The batch and time dimensions are
untouched; each integer just grew 768 numbers underneath it.

And the result really is the row, copied:

```python
>>> tok_emb[0, 0, :6]
tensor([-0.0686, -0.0203,  0.0645, -0.0621, -0.1135, -0.0623])
>>> wte.weight[464][:6]
tensor([-0.0686, -0.0203,  0.0645, -0.0621, -0.1135, -0.0623])
>>> torch.equal(tok_emb[0, 0], wte.weight[464])
True
```

That is the whole operation. `idx[0][0]` is 464, so `tok_emb[0][0]` is row 464.

It is often described with a one-hot vector: a row of 50257 numbers, all zero
except a single 1 at index 464. Multiply that by `wte` and every row is
multiplied by zero except row 464, which survives unchanged. The arithmetic
gives the same answer, and it is a useful way to see that a lookup is a matrix
multiply in disguise. No implementation builds that vector, because it would be
50257 numbers to express one index.
## x, the vector flowing through the blocks

After the embedding, position information is added and 12 blocks run:

```python
tok_emb = self.transformer.wte(idx)   # (2, 3, 768)
pos_emb = self.transformer.wpe(pos)   # (3, 768), one row per position
x = tok_emb + pos_emb

for block in self.transformer.h:
    x = block(x)
x = self.transformer.ln_f(x)
```

`x` keeps the shape `(2, 3, 768)` the entire way. One 768-vector per position,
updated repeatedly.

<div style="margin: 1.5em 0;">
  <img src="/images/wte-lmhead/forward-pass.svg"
       alt="Vertical flow of the forward pass with shapes on every arrow. idx (2, 3) int64, two sequences of three token ids, into wte(idx) which looks up one row per id, giving tok_emb (2, 3, 768). Then plus wpe(pos), position rows (3, 768), giving x (2, 3, 768). Then twelve blocks which do not change the shape. Then lm_head(x), which scores x against all 50257 rows, giving logits (2, 3, 50257), then softmax. A dashed purple line on the left connects wte and lm_head, labelled the same 50257 x 768 tensor."
       style="width: 100%; height: auto; display: block;"/>
</div>

The important part is what happens to the *content* of that vector. Take the
last position of sequence 0, the one holding `" is"`, and measure how similar it
stays to `wte[318]`, the embedding row for `" is"` itself:

```
                       cosine to wte[318]    vocab mean    rank of " is"
after the embedding           +0.4810          +0.0656          1
after block 4                 +0.0256          -0.1005        132
after block 8                 +0.0434          -0.1289         61
after block 12, pre ln_f      +0.1550          -0.2283         22
after ln_f                    -0.1722          -0.1243      50151
```

The raw cosine is easy to misread, which is why the other two columns are there.
`ln_f` shifts every row's cosine at once, so *all* 50257 come out negative after
it, and the vocabulary mean is `-0.1243`. The `-0.1722` only means something
relative to that mean. The honest measure is the last column: the rank of
`" is"` among all 50257 rows, which no global shift can change.

Read that way the story is clear. At the embedding, `" is"` is the single
nearest row out of 50257, because the vector *is* that row. By the time the
vector reaches `lm_head` it has fallen to rank 50151, the bottom 0.2% of the
vocabulary. It has stopped describing the token at that position and become
something else: a description of what should come next.
## How an output projection works

At the output the model has 768 continuous numbers and no index. It has to work
out which of 50257 tokens comes next, so it compares `x` against every row:

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

```python
>>> logits = model.lm_head(x)
>>> logits.shape
torch.Size([2, 3, 50257])
```

One score per vocabulary entry, at every position. Those scores are called
logits.

Each one is a dot product between `x` and one row of `wte`:

```python
>>> x_last = x[0, -1]           # the 768-vector after "The sky is"
>>> x_last @ wte.weight[4171]   # 4171 is " blue"
tensor(-94.34)
>>> logits[0, -1, 4171]
tensor(-94.34)
```

Same number. Doing all 50257 of those dot products at once is exactly a matrix
multiply, which is what the layer performs:

```python
>>> torch.allclose(logits[0, -1], x_last @ wte.weight.T, atol=1e-3)
True
```

<div style="margin: 1.5em 0;">
  <img src="/images/wte-lmhead/scoring.svg"
       alt="A vector x of shape (768,), the final x at the last position of The sky is, on the left, with arrows to rows of wte on the right: row 262 the scoring -93.77, row 4171 blue scoring -94.34, row 318 is scoring -102.04, row 3797 cat scoring -103.84, and 50253 more rows. Text reads logit for row i equals dot(x, wte[i]), and doing all 50257 at once is one matrix multiply, logits = x @ wte.T."
       style="width: 100%; height: auto; display: block;"/>
</div>

### What the dot product is actually asking

Mechanically that is all it is, 50257 dot products. The interesting question is
what a dot product *means* here, and the answer depends entirely on what `x` has
become by this point.

`x` started as row `wte[318]`, the embedding of `" is"`, because that is the
token at this position. Twelve blocks later it is not that any more. Attention
has pulled in `"The"` and `" sky"`, the MLPs have rewritten it repeatedly, and
what remains is not a description of `" is"`. It is a description of whatever
should come after `" is"`. The position no longer holds a token, it holds a
request.

The dot product is how that request gets matched against the vocabulary. Each of
the 50257 rows is a vector for one token, sitting in the same 768-dimensional
space as `x`, so asking `dot(x, W[i])` is asking how much of row `i` points the
same way as `x`. Geometrically that is a projection: the score is the length of
row `i`'s shadow when it is cast onto the direction `x` is facing.

<div style="margin: 1.5em 0;">
  <img src="/images/wte-lmhead/query-shadow.svg"
       alt="A geometric sketch. A thick orange horizontal arrow labelled x, the description of what fits next, after 12 blocks at the last position of The sky is. Five vocabulary vectors leave the same origin at increasing angles above it: the at p 0.152, blue at p 0.086 and falling at p 0.073 in green at shallow angles, then is at p 0.000 and cat at p 0.000 in red at wide angles. Each drops a dashed perpendicular onto the x axis and casts a thick shadow segment along it, long for the shallow green vectors and short for the wide red ones. A caption reads the shadow on x is the score, dot(x, W[i]). A legend reads same direction, long shadow, high score; wide angle, short shadow, low score. A footnote notes this is a 2D sketch of a 768D geometry with angles exaggerated."
       style="width: 100%; height: auto; display: block;"/>
</div>

So the output projection is a similarity test between one query and the whole
vocabulary. Rows aligned with `x` get long shadows and high logits, rows pointing
elsewhere get short ones. Whatever the model wants next, it expresses by pointing
`x` in the direction of the tokens that would satisfy it.

At this position, the rows that score well are:

```
dot(x, wte[  262]) " the"   =  -93.77
dot(x, wte[ 4171]) " blue"  =  -94.34
dot(x, wte[  318]) " is"    = -102.04
dot(x, wte[ 3797]) " cat"   = -103.84
```

`" is"` is the token sitting at that position, and it scores badly. `" blue"`
is not there at all, and it scores near the top. That is the clearest sign that
`x` has stopped describing its own token: the vector is furthest from the thing
it started as. After a softmax:

```
"The sky is"  ->  " the" 0.152, " blue" 0.086, " falling" 0.073
"I ate an"    ->  " apple" 0.105, " entire" 0.083, " egg" 0.082
```

One caveat worth stating, because "similarity" invites it. This is not cosine
similarity. A dot product is `|x| |w| cos(angle)`, so row length counts too, and
the rows differ a lot, from 2.45 to 6.32. Sorting the vocabulary by raw cosine
against this same `x` gives:

```
by logit    " the", " blue", " falling", " a", " full", " dark"
by cosine   "SPONSORED", "theless", "soDeliveryDate", "Reviewer", ...
```

Not one token in common in the top 100. The projection is a similarity test in
the loose sense of pointing the same way, but the length of each row is part of
the score, and the model uses it.

So the same table is being used two different ways. Going in, an index selects a
row. Coming out, a vector is compared against every row.

<div style="margin: 1.5em 0;">
  <img src="/images/wte-lmhead/two-ends.svg"
       alt="Two panels. Left: wte, nn.Embedding(50257, 768), takes a token id, one integer, does a row lookup, returns 768 numbers. Right: lm_head, nn.Linear(768, 50257, bias=False), takes 768 numbers, scores every row, returns 50257 numbers, one logit per vocabulary entry."
       style="width: 100%; height: auto; display: block;"/>
</div>
## Why there is no transpose for the unembedding

The assignment `wte.weight = lm_head.weight` only works if the two shapes
already agree. It is worth being slow about why they do, because the natural
guess is that they should not.

Start with what each side needs. `wte` is a table of 50257 rows, one per
vocabulary entry, each row 768 numbers long. So `wte.weight` is
`(50257, 768)`, and `wte(idx)` reads row `idx` out of it.

`lm_head` is `nn.Linear(768, 50257, bias=False)`. It takes 768 numbers in and
produces 50257 numbers out. Written as plain matrix multiplication, that is a
row vector of length 768 times a matrix of shape `(768, 50257)`:

```
  (1, 768)  @  (768, 50257)  ->  (1, 50257)
     x            matrix           logits
```

The inner dimensions have to match, 768 against 768, and the outer ones survive.
So the matrix that the multiplication wants is `(768, 50257)`, which is the
transpose of what `wte` wants. That is the mismatch you would expect, and it is
why the missing transpose looks suspicious.

It is not there because PyTorch does not store the matrix in that orientation.
`nn.Linear` stores its weight as `(out_features, in_features)`, not
`(in_features, out_features)`:

```python
>>> lm_head = nn.Linear(768, 50257, bias=False)
>>> lm_head.weight.shape
torch.Size([50257, 768])
>>> wte.weight.shape
torch.Size([50257, 768])
```

Identical. Row `i` of `lm_head.weight` is the 768-long vector that produces
output number `i`, which is the same thing row `i` of `wte.weight` is: the
vector belonging to token `i`. Both are "one row per vocabulary entry", so both
are `(50257, 768)`.

The transpose still has to happen for the multiply to be legal. It happens when
the layer runs, not in storage. `nn.Linear.forward` calls `F.linear(x, W)`,
which computes `x @ W.T`:

```python
>>> x.shape                       # (b, t, n_embd)
torch.Size([2, 3, 768])
>>> W = lm_head.weight
>>> W.shape
torch.Size([50257, 768])
>>> W.T.shape
torch.Size([768, 50257])
>>> lm_head(x).shape
torch.Size([2, 3, 50257])
>>> torch.allclose(lm_head(x), x @ W.T)
True
```

So the shapes line up like this, with the last axis of `x` contracting against
the 768 axis and the 50257 axis becoming the new last axis:

```
  x         (2, 3,  768)
  W.T             (768, 50257)
                   ^^^^  contracted
  logits    (2, 3,       50257)
```

`b` and `t` are carried through untouched, exactly as they were in the lookup.
The lookup turned each integer into 768 numbers; the unembedding turns each 768
numbers back into 50257 scores. Same two leading axes, different trailing one.

And `W.T` is not a second copy of the matrix. Transposing in PyTorch returns a
view with the strides swapped, over the same memory:

```python
>>> W.stride(), W.T.stride()
((768, 1), (1, 768))
>>> W.is_contiguous(), W.T.is_contiguous()
(True, False)
```

So there are three separate things that are easy to run together, and only one
of them is real storage. The stored matrix is `(50257, 768)`. The transposed
view `(768, 50257)` costs nothing and exists only for the duration of the
multiply. And `wte` never transposes at all, since a row lookup does not care
about the second axis.

That is the whole answer. There is no mismatch for the assignment to fix,
because the orientation the two uses need is the same one, and after the
assignment there is a single tensor:

```python
>>> wte.weight is lm_head.weight
True
>>> wte.weight.data_ptr() == lm_head.weight.data_ptr()
True
>>> wte.weight.numel()
38597376
```

<div style="margin: 1.5em 0;">
  <img src="/images/wte-lmhead/one-buffer.svg"
       alt="A central purple box labelled one tensor, (50257, 768), 38,597,376 floats, one data_ptr(). On the left a blue box labelled wte reads row i, no transpose. On the right an orange box labelled lm_head computes x @ W.T, transpose at call time. Both arrows point into the same central box."
       style="width: 100%; height: auto; display: block;"/>
</div>

Sharing it saves 38,597,376 parameters, which is 154.4 MB in fp32. It is also
where the "124M" in GPT-2 124M comes from: tied the model is 124,439,808
parameters, untied it would be 163,037,184.

## Where this leaves us

So far this is all mechanics. `idx` is `(b, t)` integers, `wte` turns each into a
row giving `(b, t, 768)`, twelve blocks rewrite the contents without changing the
shape, and `lm_head` compares each vector against all 50257 rows giving
`(b, t, 50257)` logits.

The shapes agree. `nn.Linear` stores its weight as `(out_features, in_features)`,
which for `lm_head` is `(50257, 768)`, the same as `wte`. Both are one row per
vocabulary entry. The transpose happens at call time, inside `F.linear`, and it
is a view rather than a copy. Nothing has to be rearranged for the assignment to
work.

But legal is not the same as sensible. A row lookup and a similarity comparison
are not obviously the same job, and the vector arriving at `lm_head` is supposed
to describe the *next* token while the row it looks up describes the *current*
one. That is the subject of
[part 2](/blog/2026-08-23/weight-tying-2-one-matrix-two-meanings.html).
