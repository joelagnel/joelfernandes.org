---
layout: post
title: "Teaching a Bigram Model Where It Is: Positional Encoding in makemore"
date: 2026-06-29
categories: [ml, language-models]
tags: [makemore, karpathy, positional-encoding, bigram, neural-networks, python, pytorch]
author: Joel Fernandes
published: true
---

> **Context:** This post follows Andrej Karpathy's
> [makemore](https://www.youtube.com/watch?v=TCH_1BHY58I) (Part 2 of the
> spelled-out intro series) — the intro to language modeling. The positional-encoding twist below
> is my own addition on top of his bigram model. If you haven't watched it
> yet, it's excellent. Part 3 (MLP) is the natural next step after this:

<div style="margin: 1.5em 0 2em 0; text-align: center;">
  <a href="https://www.youtube.com/watch?v=PaCmpygFfXo" target="_blank" rel="noopener" style="display:inline-block; text-decoration:none;">
    <img src="https://img.youtube.com/vi/PaCmpygFfXo/maxresdefault.jpg"
         alt="Building makemore Part 2: MLP — Andrej Karpathy"
         style="width:100%; max-width:600px; border-radius:8px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);">
    <div style="margin-top:0.5em; font-size:0.95em; color:#555;">
      ▶ <strong>Building makemore Part 2: MLP</strong> — Andrej Karpathy &nbsp;·&nbsp; 1.1M views
    </div>
  </a>
</div>

*How adding a "position-in-the-word" signal to Karpathy's neural bigram model
cut the loss by ~6.7%.*

---

## Table of Contents

- [Background](#background)
  - [The dataset and bigrams](#the-dataset-and-bigrams)
  - [The neural bigram model](#the-neural-bigram-model)
  - [The baseline loss](#the-baseline-loss)
- [The idea: a bigram model is blind to position](#the-idea-a-bigram-model-is-blind-to-position)
- [The implementation](#the-implementation)
  - [Augmenting the input vector](#augmenting-the-input-vector)
  - [Growing the weight matrix](#growing-the-weight-matrix)
- [Results](#results)
- [Why it works](#why-it-works)
- [Sampling with positions](#sampling-with-positions)
- [Caveats and next steps](#caveats-and-next-steps)
- [Conclusion](#conclusion)

---

## Background

This post follows Andrej Karpathy's excellent
["The spelled-out intro to language modeling: building makemore"](https://www.youtube.com/watch?v=TCH_1BHY58I)
(Part 2 of the series). The positional-encoding twist described here is my own addition on top
of his bigram model.

If you've already watched the video, skip ahead to
[The idea](#the-idea-a-bigram-model-is-blind-to-position). Otherwise, here's the
minimum you need.

### The dataset and bigrams

`makemore` learns to generate new names from a list of ~32,000 real names
(`names.txt`). Each name is wrapped with a special boundary token `.` and broken
into **bigrams** — pairs of adjacent characters. For the name `emma`:

```
.e   e m   m m   m a   a .
```

We map each of the 27 tokens (`.` + `a`–`z`) to an integer with `stoi`/`itos`,
and build two parallel tensors: `xs` (the current character) and `ys` (the next
character we want to predict). Over the whole dataset that's **228,146** bigram
examples.

### The neural bigram model

Instead of just counting bigram frequencies, Karpathy frames the bigram model as
a tiny one-layer neural net:

1. **One-hot encode** the input character into a 27-dim vector.
2. **Matrix-multiply** by a weight matrix `W` of shape `(27, 27)` to get
   *logits* (log-counts).
3. **Softmax** the logits into a probability distribution over the next
   character.

```python
xenc   = F.one_hot(xs, num_classes=27).float()  # (N, 27)
logits = xenc @ W                                # (N, 27)
counts = logits.exp()
probs  = counts / counts.sum(1, keepdims=True)   # softmax -> (N, 27)
```

We train `W` by minimizing the **average negative log-likelihood** (NLL) of the
true next character, with a little L2 regularization, using plain gradient
descent:

```python
loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
loss.backward()
W.data += -50 * W.grad
```

### The baseline loss

After 200 steps of gradient descent, the baseline bigram model settles at:

```
199  2.4829957485198975
```

So our number to beat is **loss ≈ 2.483**.

---

## The idea: a bigram model is blind to position

A pure bigram model makes the next-character prediction using **only the current
character**. Feed it an `a` and it always returns the exact same distribution,
whether that `a` is the first letter of the name or the seventh.

But position clearly matters in names:

- The **first** character (right after `.`) follows a very different
  distribution than a mid-word character — some letters are common name-starters,
  others almost never start a name.
- The probability of ending the word (emitting `.`) grows the **further into the
  word** we are. A 1-letter name is rare; after 6–7 characters the name is very
  likely to end soon.

A vanilla bigram simply cannot express "I'm at position 5, so `.` is now more
likely." So the idea: **tell the model where in the word each character sits**,
and let it learn position-dependent behavior.

This is the same intuition behind *positional encoding* in Transformers — an
attention/bigram mechanism that is otherwise order-agnostic gets an explicit
signal about position.

---

## The implementation

### Augmenting the input vector

The input to the network is just a one-hot vector. So we **concatenate a second
one-hot** that encodes the position of the character within its word. With a cap
of 23 positions (the longest names in the dataset), the input grows from 27 to
**50** dimensions:

```
[ 27-dim character one-hot | 23-dim position one-hot ]
```

For the bigrams of `emma`, the augmented encodings look like this (note the
second `1` marching across the positional block):

```
.  ->  char[0]=1                  pos[0]=1
e  ->  char[5]=1                  pos[1]=1
m  ->  char[13]=1                 pos[2]=1
m  ->  char[13]=1                 pos[3]=1
a  ->  char[1]=1                  pos[4]=1
```

### Growing the weight matrix

The only other change is that `W` grows from `(27, 27)` to `(50, 27)`. The first
27 rows are the familiar character weights; the new 23 rows are
**per-position weight vectors**. Because the matmul is linear, the logits become
the **sum of a character contribution and a position contribution**:

```python
W = torch.randn((50, 27), generator=g, requires_grad=True)

xenc   = encode_xs(xs)   # (N, 50)  <- now includes position
logits = xenc @ W        # (N, 27)
```

Everything downstream — softmax, loss, backward, update — is **byte-for-byte
identical** to the baseline. The entire modification lives in the input encoding
and the shape of `W`.

---

## Results

Same data, same training loop, same 200 steps, same learning rate. The only
difference is the positional input and the shape of `W`:

| Model                         | `W` shape | Input dim | Final loss (step 199) |
|-------------------------------|-----------|-----------|-----------------------|
| Baseline bigram               | `(27, 27)`| 27        | **2.4830**            |
| Bigram + positional encoding  | `(50, 27)`| 50        | **2.3170**            |

That's a drop of `2.4830 − 2.3170 = 0.1660`, a **~6.7% reduction in loss** — a
meaningful improvement for a model that is still, fundamentally, a single linear
layer.

The positional model's training tail:

```
195  2.3171677589416504
196  2.317129611968994
197  2.317091464996338
198  2.317054510116577
199  2.317017078399658
```

---

## Why it works

Because the matmul is linear, the augmented model computes:

```
logits = (character weights for this char) + (position weights for this position)
```

The position rows act as a **learned, position-dependent bias** added on top of
the bigram logits. With that extra freedom the model can finally express things
the vanilla bigram never could:

- At **position 0** (right after `.`), push up the logits of common
  name-starting letters.
- At **later positions**, steadily raise the logit of the end-token `.`, so long
  partial names are encouraged to terminate.

None of this required a deeper network, an embedding table, or attention — just a
richer input representation and 23 extra weight vectors.

---

## Sampling with positions

Generation has to feed the position too, since `W` now expects a 50-dim input.
During sampling we track the current length of the generated string and set the
corresponding position bit:

```python
out = []
ix  = 0
while True:
    pos = len(out)
    extra_cells = torch.zeros((1, 23))
    extra_cells[0, pos] = 1

    xenc   = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    xenc   = torch.cat((xenc, extra_cells), dim=1)   # (1, 50)
    logits = xenc @ W
    p      = logits.exp() / logits.exp().sum(1, keepdims=True)

    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
        break
```

A nice qualitative side effect: with the position signal, sampled names are
shorter and terminate more naturally (the model has learned that `.` becomes more
likely deeper into the word), instead of occasionally rambling on:

```
baseline:     momasurailezitynn.   konimittain.   ...
positional:   moman.   raile.   kayha.   konimi.   ...
```

---

## Caveats and next steps

A few honest limitations worth calling out:

- **Sampling can overflow if names exceed 23 chars.** `extra_cells[0, pos] = 1`
  would index out of bounds; in practice generated names stay short, but a clamp
  there would make it robust.
- **One-hot positions don't generalize between positions.** Position 5 and
  position 6 are as unrelated to the model as position 5 and position 20. A dense
  or sinusoidal encoding would let nearby positions share structure.

The natural next step is to watch Part 3, the third video in the series.

---

## Conclusion

Starting from Karpathy's neural bigram model, a single conceptual change — *tell
the model where each character sits in its word* — bought a **~6.7% loss
reduction** with no change to the training loop and only 23 extra weight vectors.



---

*Based on Andrej Karpathy's
[makemore Part 2](https://www.youtube.com/watch?v=PaCmpygFfXo). The positional
encoding extension and analysis are my own.*
