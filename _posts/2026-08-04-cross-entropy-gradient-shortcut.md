---
layout: post
title: "Backprop: Easy Gradient Shortcut Through Cross-Entropy"
date: 2026-08-04
categories: [machine-learning, neural-networks, backpropagation]
tags: [machine-learning, ml, deep-learning, neural-networks, neural-nets, backpropagation, backprop, autograd, chain-rule, gradients, gradient-descent, cross-entropy, softmax, logits, loss-function, classification, numpy, pytorch, calculus]
author: Joel Fernandes
description: "Part 5. The long way to the logit gradient needs eight intermediate tensors and five backward passes. Freeze everything except the one logit you care about, and it collapses to three lines."
published: false
---

> **Context:** This is part 5 in a series that comes out of Andrej Karpathy's
> [Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI), where you
> hand-write every backward pass instead of calling `.backward()`. Parts 1 through 4 worked
> through individual operations. This one is Exercise 2 from the notebook, and it is the
> exercise where all of that effort gets refunded.

<div style="margin: 1.5em 0 2em 0; text-align: center;">
  <a href="https://www.youtube.com/watch?v=q8SA3rM6ckI" target="_blank" rel="noopener" style="display:inline-block; text-decoration:none;">
    <img src="https://img.youtube.com/vi/q8SA3rM6ckI/maxresdefault.jpg"
         alt="Building makemore Part 4: Becoming a Backprop Ninja, by Andrej Karpathy"
         style="width:100%; max-width:600px; border-radius:8px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);">
    <div style="margin-top:0.5em; font-size:0.95em; color:#555;">
      &#9654; <strong>Building makemore Part 4: Becoming a Backprop Ninja</strong>, Andrej Karpathy
    </div>
  </a>
</div>

---

In the earlier posts we broke the loss calculation into its smallest atomic pieces and
backpropagated through every one of them. Karpathy's own verdict on that approach is worth
quoting, because it is the setup for this entire post:

> it basically turns out that in this first exercise we were doing way too much work, we were
> back propagating way too much, and it was all good practice and so on, but it's not what you
> would do in practice.

Here is the forward pass we had been carrying around. Eight lines, eight intermediate tensors,
every one of which needed its own backward pass:

```python
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes    # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
```

And here is the whole thing in PyTorch:

```python
loss_fast = F.cross_entropy(logits, Yb)
```

The interesting part is not that the forward pass collapses. It is that the *backward* pass
collapses too, from five separate gradient computations down to this:

```python
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
```

Three lines. If you have not seen the derivation, those lines look like magic, particularly the
middle one, which appears to subtract 1 from an arbitrary-looking set of positions. By the end
of this post each line should look inevitable.

## The block we are differentiating

Zoom out first. Cross-entropy is a box that eats two things and produces one number.

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/xent-grad/pipeline.svg" alt="The logits matrix and the Yb labels feed into a cross-entropy block which outputs a single loss value L. A dashed return path shows the gradient dL/dlogits flowing back to the logits." style="max-width: 100%; height: auto;"/>
</div>

It takes `logits`, the raw scores the network produced, and `Yb`, the characters that should
actually have come next. It returns `L`, one number measuring how likely the model was to
predict `Yb`. Low probability on the right answer gives a high loss.

What we want back out is `dL/dlogits`, which has to be the same shape as `logits`, because a
gradient always has the same shape as the thing it is the gradient of.

## Shapes

The logits matrix is 32 rows by 27 columns.

The 32 is the batch size, one row per example. The 27 is the number of output neurons, one per
character in the vocabulary: 26 letters plus the `.` token that marks the start and end of a
name.

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/xent-grad/shapes.svg" alt="A 32 by 27 grid of logits, softmaxed row by row into probabilities, with one marked cell per row plucked out into a 32 by 1 column, which is then negative-logged and averaged into a single loss." style="max-width: 100%; height: auto;"/>
</div>

Four things happen, in order:

1. **Softmax along each row.** Each row of 27 raw scores becomes 27 probabilities that sum to 1.
2. **Pluck one probability per row.** `Yb[i]` says which column is the correct character for row
   `i`, so we take `probs[i, Yb[i]]`. That gives a column of 32 numbers.
3. **Take the negative log of each.**
4. **Average over the batch.**

Which in code is exactly what that last line of the long forward pass said:

```python
loss = -logprobs[range(n), Yb].mean()
```

Step 2 is worth pausing on. Out of 32 x 27 = 864 numbers in the probability matrix, the loss
looks at exactly **32 of them**, one per row. The other 832 are never read. That fact does not
mean their gradients are zero, as we will see, but it does mean the loss expression itself is a
lot smaller than the matrix suggests.

## Why the negative log

The loss for a single example is `-log(p)`, where `p` is the probability the model assigned to
the correct character. That choice is doing something specific:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/xent-grad/neglog.svg" alt="A plot of negative log p against p. As p approaches 1 the loss approaches 0. As p approaches 0 the loss grows without bound." style="max-width: 100%; height: auto;"/>
</div>

If the model gave the right answer a probability near 1, `log(p)` is near 0 and the loss is
near 0. Nothing to fix. If it gave the right answer a probability near 0, `log(p)` heads for
minus infinity and the loss becomes enormous. Being confidently wrong is expensive, and it is
supposed to be.

Probabilities live between 0 and 1, so their logs are always negative. The minus sign out front
just flips that back to a positive loss.

## Two things to notice before any calculus

Both of these make the derivation smaller, and it is worth naming them explicitly.

**Rows are independent.** Row `i` of the loss depends only on row `i` of the logits. Softmax
runs along a row, and the plucked probability comes from that same row. Nothing crosses between
examples. So we can derive the gradient for one row, and then apply the identical formula to
all 32.

**The 1/n is just a constant.** The batch loss is the mean, so there is a factor of `1/n` out
front. Constants pass straight through differentiation, so park it, do the interesting work
without it, and multiply it back at the very end.

That leaves one clean question: for a single row, what is the derivative of the loss with
respect to one logit in that row?

## The trick: freeze everything except one

Here is the move that makes the rest easy.

Write the softmax for a single row. Call the logit we are differentiating with respect to `a`.
The softmax denominator is a sum over all 27 exponentials:

```
SM = e^a / (e^a + e^b + e^c + e^d + ...)
```

We are differentiating with respect to `a` and nothing else. Every other logit in that row is
being held fixed. So every other exponential in that denominator is a **constant**. Not
approximately, not for convenience: `e^b`, `e^c`, `e^d` and the rest genuinely do not change when
`a` changes.

Give the whole pile of them one name, `K`:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/xent-grad/k-trick.svg" alt="A row of 27 cells with one highlighted as the live variable e to the a. The remaining 26 are greyed out and braced together, labelled as the constant K." style="max-width: 100%; height: auto;"/>
</div>

```
SM = e^a / (e^a + K)
```

One variable, one constant. A 27-variable problem just became a single-variable calculus
problem, the kind you can hand to Wolfram Alpha, or do by hand in two lines.

One rearrangement will be useful shortly. Since `SM = e^a / (e^a + K)`, we can solve for the
denominator and then for `K` itself:

```
e^a + K = e^a / SM

K = e^a/SM - e^a
```

Keep that in your pocket.

## Two cases, and why there are exactly two

Now, the part that trips people up. We need `dL/da` for **every** logit in the row, all 27 of
them, not just the correct one. And the loss expression looks different depending on whether
`a` happens to be the correct character or not.

Here is the crucial bit of intuition. The loss only ever looks at the probability of the
**correct** character:

```
L = -log( e^(correct logit) / (sum of all 27 exponentials) )
```

Now ask where a given logit `a` appears in that expression.

- If `a` **is** the correct character, then `e^a` appears **twice**: once in the numerator, and
  once inside the denominator sum.
- If `a` is **not** the correct character, then `e^a` appears exactly **once**, and only in the
  **denominator**.

That second case is the one worth sitting with. A wrong character's logit still affects the
loss, even though the loss never looks at its probability. It affects the loss because it is
part of the normalizing sum. Raise a wrong logit and you inflate the denominator, which shrinks
the correct character's probability, which raises the loss. The influence is entirely indirect,
routed through the denominator.

So in the wrong-character case the numerator is a constant, and we can just call it `C`:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/xent-grad/two-cases.svg" alt="Two panels side by side. Case A, a is the correct character, gives SM minus 1. Case B, a is a wrong character, gives SM. The only difference is the minus 1." style="max-width: 100%; height: auto;"/>
</div>

### Case A: `a` is the correct character

The loss for this row, in terms of `a` and `K`:

```
L = -log( e^a / (e^a + K) )
```

Differentiating with respect to `a`. This is exactly the query I put into Wolfram Alpha:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/xent-grad/wolfram-correct.jpg" alt="Wolfram Alpha computing the derivative of minus log of e to the x over e to the x plus k, giving minus k over k plus e to the x." style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;"/>
  <div style="margin-top: 0.5em; font-size: 0.9em; color: #666;">Wolfram uses <code>x</code> for the variable and <code>k</code> for the constant. Read them as <code>a</code> and <code>K</code>.</div>
</div>

```
dL/da = -K / (K + e^a)
```

Now substitute `K = e^a/SM - e^a` from earlier, and watch it fall apart:

```
dL/da = -( e^a/SM - e^a ) / ( e^a/SM )
```

Multiply top and bottom by `SM/e^a`:

```
      = -( 1 - SM )

      = SM - 1
```

That is it. The gradient on the correct character's logit is just its own softmax probability,
minus one.

### Case B: `a` is a wrong character

Now `e^a` only appears in the denominator. The numerator is the correct character's
exponential, which does not depend on `a` at all, so it is a constant `C`:

```
L = -log( C / (e^a + K) )
```

Straight to Wolfram again:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/xent-grad/wolfram-incorrect.jpg" alt="Wolfram Alpha computing the derivative of minus log of k over e to the x plus k, giving e to the x over k plus e to the x." style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;"/>
</div>

```
dL/da = e^a / (e^a + K)

      = SM
```

Which is the softmax probability of `a`, with nothing subtracted.

Notice that Wolfram also offers the alternate form `1 - k/(k + e^x)`, which is a nice mirror of
Case A. Same expression, opposite framing.

## Merging the two cases

Put them side by side:

```
a is the correct character:   dL/da = SM - 1
a is a wrong character:       dL/da = SM
```

The `SM` term is common to both. The only difference is a `-1` that shows up exactly on the
correct character's position. So write it as one expression:

```
dL/da = SM - 1[a is the correct character]
```

where the bracket is 1 when true and 0 otherwise. And now bring back the `1/n` we parked
earlier, since the batch loss is a mean over `n` examples:

```
dL/da = ( SM - 1[a is the correct character] ) / n
```

Which is exactly the three lines of code, in the same order:

```python
dlogits = F.softmax(logits, 1)     # the SM term, for all 32 x 27 entries at once
dlogits[range(n), Yb] -= 1         # the -1, only at the correct positions
dlogits /= n                       # the 1/n from the mean
```

Three lines, three terms, nothing left over. `F.softmax(logits, 1)` runs the softmax along
dimension 1, meaning along each row. `[range(n), Yb]` is the same fancy indexing that plucked
out the probabilities in the forward pass, now used to subtract 1 at those same positions. And
the division by `n` is the gradient flowing back through the mean.

## Reading the answer as forces

There is a genuinely nice way to look at the result, and it is the best part of Karpathy's
explanation.

Take one row of `dlogits`, multiply it by `n` to undo the batch scaling, and read each cell not
as a number but as a force acting on that logit.

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/xent-grad/forces.svg" alt="A bar chart of one row of the gradient. One long red bar below the axis at the correct character, 26 short blue bars above the axis at the wrong characters. The row sums to zero." style="max-width: 100%; height: auto;"/>
</div>

The correct character gets `SM - 1`, which is always negative, since `SM` is a probability below
1. Negative gradient means gradient descent will push that logit **up**. Every wrong character
gets `SM`, always positive, so those logits get pushed **down**.

Now add up the whole row. The probabilities sum to 1, and we subtracted exactly one 1, so:

```
sum of the row = 1 - 1 = 0
```

Every row of `dlogits` sums to exactly zero. The upward pull on the correct character and the
total downward push on the other 26 are always perfectly balanced. The gradient never lifts or
lowers a row as a whole, it only moves probability mass around within the row.

The magnitudes are just as readable. The size of each force is exactly the probability the
model put in the wrong place:

- A **confidently wrong** prediction, where some wrong character got probability 0.9, gets a
  hard shove downward, and the correct character gets an equally hard pull upward.
- A **perfect** prediction, probability 1 on the correct character and 0 everywhere else, gives
  `1 - 1 = 0` at the correct position and `0` everywhere else. The entire row is zeros. No
  force, nothing to fix.

So the amount by which you mispredict is exactly the strength of the correction. Karpathy's
image for this is a pulley system: you are up at the logits pulling on the correct answer and
pushing down the wrong ones, and that tension translates back through the network until it
finally tugs on the weights and biases. Each update, the parameters give in to the tug a little.
That is training.

## Verifying it

Never trust a hand-derived gradient you have not checked:

```python
loss_fast = F.cross_entropy(logits, Yb)
print(loss_fast.item(), 'diff:', (loss_fast - loss).item())

dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n

cmp('logits', dlogits, logits)
```

The comparison comes back approximate rather than exactly equal, with a maximum difference
around **5e-9**. That is floating point noise, not a mistake in the derivation. The long chain
and the short expression are mathematically identical but execute a different sequence of
operations, so they round differently in the last bits.

If anything, the short form is the better-behaved one. The eight-line version computes
exponentials, sums them, inverts the sum, multiplies, and takes a log, with rounding error
accumulating at each step. The three-line version does none of that intermediate work.

## Wrapping up

This is the payoff for parts 1 through 4. All that careful hand-differentiation of individual
operations was the training; this is the thing it trained you to spot.

The shortcut exists for two reasons, and both are worth carrying forward:

- **The loss only touches one column per row.** Out of 864 numbers, the loss expression reads
  32. That keeps the expression you have to differentiate small.
- **Freezing 26 of the 27 logits turns matrix calculus into ordinary calculus.** The `K`
  substitution is not an approximation. Those other logits genuinely are constants with respect
  to the one you are differentiating, and naming them collectively is what reduces the problem
  to something you can do on paper.

The next time a backward pass in your framework looks suspiciously short, it is probably because
somebody did this on paper first.
