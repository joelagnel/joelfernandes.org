---
layout: post
title: "Weight tying, part 2: how one matrix means two different things"
date: 2026-08-23
categories: [machine-learning, neural-networks, transformers]
tags: [machine-learning, ml, deep-learning, neural-networks, transformers, gpt-2, llm-c, embeddings, weight-tying, pytorch]
author: Joel Fernandes
description: "The same 50257 vectors serve a lookup at the input and a prediction at the output. Why that is not a contradiction, measured on real GPT-2 weights."
published: true
---

This is part 2 of three on weight tying.
[Part 1](/blog/2026-08-23/weight-tying-1-two-ends.html) worked through what `wte` and `lm_head` each do and showed that
their shapes agree, so this line in llm.c's `train_gpt2.py` is legal:

```python
self.transformer.wte.weight = self.lm_head.weight
```

Legal, but strange. `wte` maps a token to a vector. `lm_head` maps a vector to
the *next* token. Those are not the same mapping, and one matrix is doing both.
This post is about why that works.

As before, every number is printed from the real GPT-2 124M weights.

## How one matrix does two different jobs

The shapes agreeing explains why the assignment is legal. It does not explain
how the same numbers serve a lookup at one end and a comparison at the other,
which is the part that still felt like a coincidence to me.

What resolved it was to stop thinking of `wte.weight` as an operation. It is not
a lookup and it is not a projection. It is a set of 50257 vectors, one per
token, each 768 numbers long. Reading one of them out and scoring a query
against all of them are two different things you can do with a set of vectors,
and neither is more natural than the other.

The difference between the two is which axis you move along:

```
lookup      W[i]       index the 50257 axis   ->  (768,)      one row
unembed     x @ W.T    sum over the 768 axis  ->  (50257,)    one score per row
```

Going in you have an index and no vector, so you travel down the vocabulary axis
and pull one row out whole. Coming out you have a vector and no index, so you
travel along the 768 axis instead, collapsing each row to a single number, its
dot product with `x`:

```python
>>> W[i].shape                                  # index the 50257 axis
torch.Size([768])
>>> (x @ W.T).shape                             # sum over the 768 axis
torch.Size([50257])
>>> torch.allclose((x @ W.T)[i], torch.dot(x, W[i]))
True
```

<div style="margin: 1.5em 0;">
  <img src="/images/wte-lmhead/two-contractions.svg"
       alt="Two panels sharing the same purple W of shape (50257, 768). Left, labelled lookup: the 50257 axis is marked as indexed, one row is highlighted and pulled out, giving W[i] of shape (768,), one row of the table, and the other 50256 rows are untouched. Right, labelled unembedding: x of shape (1, 768) multiplies W.T of shape (768, 50257), a view and not a copy, the 768 axis is marked as summed, giving shape (1, 50257), one score per row, every row read."
       style="width: 100%; height: auto; display: block;"/>
</div>

The two are not symmetric in what they throw away. The lookup keeps one row
intact and ignores the other 50256, so it is exact and reversible. The
unembedding reads every row but crushes each one to a scalar, so 768 numbers
become a single number and everything about the direction within that row is
gone. That asymmetry is why the output side needs a softmax and a loss before it
means anything, while the input side is just a table read.
## Why x ends up predicting the next token, not the current one

This is the part that looks impossible. `wte` maps a token to a vector.
`lm_head` maps a vector to the *next* token. Same matrix, and those are not the
same mapping.

The resolution is the drift measured in
[part 1](/blog/2026-08-23/weight-tying-1-two-ends.html): by the last block, `x`
is no longer a representation of the token at that position. It is a description of
what fits next. So the dot product is not comparing one token to another. It is
asking how well the token at row `i` answers what this position is looking for.

Training is what makes it so, and the mechanism is worth walking through slowly,
because it is the reason one matrix can hold both meanings at once.

Picture one training step on `"The sky is"`, where the correct next token is
`" blue"`. The model produced `x`, scored every row against it, and got a
probability for each token. `" blue"` came out at 0.086, so the model was
mostly wrong and there is a correction to apply.

Cross entropy applies that correction in two places at once. It nudges the
**rows** and it nudges **`x`**, and the two nudges point at each other:

- the row for `" blue"` moves **toward `x`**, because that would have raised its
  score
- every other row moves **away from `x`**, because that would have lowered theirs
- `x` moves **toward the row for `" blue"`**, for the same reason

That is the entire mechanism. The target row and the vector that failed to
predict it are dragged together from both ends, a little on every step where
that token is the answer. Do it across a corpus and the row for `" blue"` ends
up sitting where the `x` vectors that should predict `" blue"` tend to point.

<div style="margin: 1.5em 0;">
  <img src="/images/wte-lmhead/the-pull.svg"
       alt="A sketch of embedding space with an orange point labelled wte[4171], the row for the token blue, and a green point labelled h, the final state after The sky is. A green arrow runs from h toward the orange point labelled minus dL by dh points at the target row, and an orange arrow runs from the row back toward h labelled minus dL by dW[4171] points at h. A grey point labelled cat has a dashed arrow pointing away, labelled every other row is pushed the other way."
       style="width: 100%; height: auto; display: block;"/>
</div>

The pushes are not equal, and the sizes are the interesting part. How hard a row
gets pushed away is proportional to the probability the model gave it, so the
only rows that feel anything are the ones that were plausible. On this example
`" the"` had probability 0.152 and gets shoved hard, while `" cat"` had 0.0000064
and is left essentially untouched:

```
" the"   p = 0.15240622    pushed with 17% of the force pulling " blue" in
" cat"   p = 0.00000641    pushed with 0.0007% of it
```

88% of the vocabulary has probability under one in a million at this position, so
almost the whole table is ignored on this step. The correction is spent on the
handful of tokens that were genuinely in contention, which is what you would want
and also why training does not simply blow the table apart.

If you want the calculus, it is three lines. With `p` the predicted probabilities
and `t` the correct token:

```
dL/dW[t]  =  (p_t - 1) * x        the target row
dL/dW[i]  =   p_i * x             every other row
dL/dx     =  sum_i p_i * W[i]  -  W[t]
```

The signs are the whole story. `p_t` is a probability so `(p_t - 1)` is negative,
which makes the step `-dL/dW[t]` a positive multiple of `x`: the target row moves
toward `x`. Every other `p_i` is positive, so those rows move the opposite way.
And `-dL/dx` contains `+W[t]`, so `x` moves toward the target row.

Measured on the real model, with `" blue"` as the target:

```
cos(step for " blue" row,  x)          =  1.000    exactly toward x
cos(step for " cat" row,   x)          = -1.000    exactly away from x
cos(step for x,  " blue" row)          =  0.845    mostly toward that row
```

The first two are exact because those steps are literally a scalar times `x`. The
third is 0.845 rather than 1.0 because `x`'s step is `W[t]` minus a
probability-weighted average of all the rows, and that average is not small on a
trained model. So `x` moves mostly toward `" blue"`, tilted away from wherever
the rest of its probability mass was sitting.

That is the answer to the section title. Cross entropy drives the vector that
*predicts* a token and the vector that *represents* that token to the same place.
The two uses are not conflicting jobs fighting over one matrix, they are one
constraint applied from both ends. This is why sharing the matrix works at all,
and it is the argument Press and Wolf made when they proposed the technique in
[Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859).
## Watching " dog" stop being an animal, block by block

The cleanest demonstration I found is `"The dog"`, because it separates two
things that sound the same. `" cat"` is one of the closest tokens to `" dog"` in
the table. It is also a terrible prediction for what comes after `"The dog"`.
Since both facts come out of the same matrix, they show the two uses pulling
apart on a single row.

Ask the table who `" dog"`'s neighbours are, by cosine between rows:

```
" dogs"    0.796
" Dog"     0.734
" canine"  0.652
" puppy"   0.621
" pet"     0.551
" cat"     0.550      rank 9 of 50257
```

That is a meaning neighbourhood, and `" cat"` sits right in it. Now ask the same
matrix what follows `"The dog"`:

```
" was"   0.143
"'s"     0.104
","      0.085
" is"    0.073
" had"   0.048

" cat"   0.00015     rank 423
" dog"   0.00040     rank 199
```

<div style="margin: 1.5em 0;">
  <img src="/images/wte-lmhead/dog-slot.svg"
       alt="Two panels. Left, in the table: nearest rows to wte for the token dog, listing dogs 0.796, Dog 0.734, canine 0.652, puppy 0.621, pet 0.551, and cat 0.550 highlighted in green, annotated cat is a close neighbour, rank 9 of 50257. Right, at the output: what follows The dog, as a bar chart listing was 0.143, apostrophe s 0.104, comma 0.085, is 0.073, had 0.048, and cat 0.00015 highlighted in red with a nearly invisible bar, annotated cat is a bad answer, rank 423 of 50257. Caption reads right meaning, wrong slot."
       style="width: 100%; height: auto; display: block;"/>
</div>

Every one of the top answers is a verb, a possessive or a punctuation mark.
Nothing that could be an animal is anywhere near the top, and `" cat"` has
dropped from rank 9 to rank 423. `" dog"` itself is at rank 199, so the model is
also not predicting the token that is literally sitting there.

This is the drift made concrete. `" cat"` is the right *meaning* and the wrong
*slot*. English has just committed to a noun phrase, so what is needed next is
whatever continues a subject, and that is grammatical, not semantic. A second
animal in that position would be nonsense.

You can watch the switch happen block by block by unembedding the intermediate
`x` at the `" dog"` position, which is the logit lens trick from
[nostalgebraist](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens):

```
embedding   " dog" 0.997,  " dogs" 0.003,  " canine" 0.000
block  2    " dog" 0.937,  " dogs" 0.034,  "Dog"     0.009
block  4    " dog" 0.908,  " dogs" 0.017,  " catcher" 0.010
block  6    " dog" 0.631,  "fight" 0.067,  "'s"      0.063
block  8    "'s"   0.366,  " dog"  0.140,  " catcher" 0.087
block 10    "'s"   0.314,  " dog"  0.099,  " was"    0.065
block 11    "'s"   0.248,  " was"  0.120,  " is"     0.064
block 12    " was" 0.143,  "'s"    0.103,  ","       0.085
```

At the embedding the vector is the token, and unembedding it just returns
`" dog"` with probability 0.997, which it has to, since the vector *is* row
`" dog"` and no row is closer to itself. Through the early blocks it stays there,
and what changes is the runner-up: `" dogs"`, then `" catcher"`, then `"fight"`,
the model filling in what kind of dog context this might be. Somewhere around
block 8 the identity gives way, `"'s"` takes the top slot, and from there it is
grammar the rest of the way.

The same thing in one number, the rank of `" dog"`'s own row by cosine to `x`:

```
embedding   rank    1
block  4    rank   12
block  8    rank   70
block 11    rank  268
```

It starts as the single nearest row out of 50257 and walks away. Nothing deletes
the token's identity; the residual stream just accumulates enough else that
identity stops being what the vector mostly is.

Which is the answer to how one matrix serves both ends. It is not being asked
the same question twice. Going in, the question is which row is this token.
Coming out, the question is which row answers this position. `" cat"` scores
well on the first and badly on the second, and both scores are read off the same
50257 vectors.

## Where this leaves us

The matrix is best thought of as a set of 50257 vectors rather than as an
operation. The lookup indexes the 50257 axis and pulls one vector out whole. The
unembedding sums over the 768 axis and crushes every vector to a single score.
Same numbers, different axis.

The reason the second one predicts the *next* token is that `x` is no longer the
current one. Cross entropy drags the target row and the vector that failed to
predict it toward each other, over and over, until the row for a token sits where
the vectors that should predict it tend to point. Measured on `"The dog"`,
`" cat"` is the 9th nearest row in the table and the 423rd best prediction. Right
meaning, wrong slot.

That is why one matrix can do both jobs. What it costs to make it do both, and
why models above about 8B stop asking, is
[part 3](/blog/2026-08-23/weight-tying-3-what-it-costs.html).
