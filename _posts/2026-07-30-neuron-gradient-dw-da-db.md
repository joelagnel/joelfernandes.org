---
layout: post
title: "Backprop: Propagating Gradient Through a Neuron"
date: 2026-07-30
categories: [machine-learning, neural-networks, backpropagation]
tags: [machine-learning, ml, deep-learning, neural-networks, neural-nets, backpropagation, backprop, autograd, chain-rule, gradients, gradient-descent, matrix-calculus, linear-layer, transpose, bias, batch, numpy, pytorch, calculus]
author: Joel Fernandes
description: "Part 3. A linear layer computes N = A @ W + b. Working out dW, dA and db by hand shows why the transpose appears, and why the bias gradient is a column sum."
published: false
---

> **Context:** Like [part 1]({% post_url 2026-07-30-broadcast-subtraction-gradient %}) and
> [part 2]({% post_url 2026-07-30-max-gradient-one-hot %}), this comes out of Andrej Karpathy's
> [Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) (Part 4 of the
> spelled-out intro series), where you hand-write every backward pass instead of calling
> `.backward()`. These notes are the longer explanation I want on hand the next time I go
> back to the video. If you have not watched it, start there:

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

The first two posts took apart single operations. This one does the thing those operations are actually building toward: a linear layer, the workhorse of every neural network.

```python
N = A @ W + b
```

Everyone can recite the answers here. `dW = A.T @ dN`, `dA = dN @ W.T`, `db = dN.sum(0)`. The transposes look arbitrary the first time you meet them, and most people end up memorizing which side the `.T` goes on. That is a shame, because if you write one small case out by hand, the transpose stops being a rule to memorize and becomes the only thing that could possibly have happened.

## The setup

Two inputs, two neurons, and a batch of two samples. The batch matters: with a single sample the bias gradient looks trivial and you miss the entire point of it.

`A` holds the inputs, one row per sample:

```
A = | a1  a2 |     <- sample 1
    | a3  a4 |     <- sample 2
```

`W` holds the weights, where `W_ij` is the weight from input `i` into neuron `j`:

```
W = | W11  W12 |
    | W21  W22 |
```

`b` is one bias per neuron, a single row that gets broadcast down to every sample:

```
b = [ b1  b2 ]
```

Multiplying out `N = A @ W + b` gives a 2x2, one row per sample:

```
N1 = a1·W11 + a2·W21 + b1        N2 = a1·W12 + a2·W22 + b2      <- sample 1
N3 = a3·W11 + a4·W21 + b1        N4 = a3·W12 + a4·W22 + b2      <- sample 2
```

Here is that same computation as a picture, with both samples drawn out:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/neuron-grad/fwd.svg" alt="Two panels. Sample 1: inputs a1 and a2 connect to neurons N1 and N2 through four labeled weights W11, W12, W21, W22, with biases b1 and b2 added. Sample 2: inputs a3 and a4 connect to N3 and N4 through the same four weights and the same two biases." style="max-width: 100%; height: auto;"/>
</div>

Two things to notice, and they are the two things the whole post rests on.

First, within a sample, every input reaches every neuron. `a1` is used twice, once per neuron.

Second, and easier to miss: the two panels are not two different layers. They are the *same* layer run on a second sample. `W11` appears in both panels. So does `b1`. `N1` and `N3` are the same neuron looking at different inputs, and `N2` and `N4` are the same neuron too.

So `W` and `b` are shared across samples, while each `a` belongs to exactly one sample. Everything that follows is a consequence of that split.

Backprop hands us the upstream gradient, matching `N` in shape:

```
dN = | dN1  dN2 |
     | dN3  dN4 |
```

and we owe three things back: `dW`, `dA` and `db`.

Every one of them comes from the same question, the one that drove both earlier posts: **where does this variable appear in the output, and how many times?**

## dW: summing over the batch

Take `W11`. Search the four expressions above for it. It shows up twice:

- in `N1`, multiplied by `a1`
- in `N3`, multiplied by `a3`

Two appearances, two paths to the loss, so the chain rule adds them:

```
dW11 = dN1·a1 + dN3·a3
```

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/neuron-grad/dw.svg" alt="Backward diagram: gradients dN1 from sample 1 and dN3 from sample 2 both flow into dW11, scaled by a1 and a3 respectively, and are summed." style="max-width: 100%; height: auto;"/>
</div>

That sum is over **samples**. It has to be. A weight is shared by the whole batch, so every sample gets an opinion about how it should change, and the gradient is all of those opinions added together.

Do the other three the same way:

```
dW11 = dN1·a1 + dN3·a3        dW12 = dN2·a1 + dN4·a3
dW21 = dN1·a2 + dN3·a4        dW22 = dN2·a2 + dN4·a4
```

Now stare at the pattern. Every entry pairs a column of `A` with a column of `dN`, summing over the sample index. That is precisely a matrix product with `A` transposed:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/neuron-grad/matrices.svg" alt="Matrix figure showing A transpose times dN equals dW, with each entry of dW written out as a sum over the two samples." style="max-width: 100%; height: auto;"/>
</div>

```
dW = A.T @ dN
```

The transpose is not a trick. `A` is indexed `[sample, input]` and `dN` is indexed `[sample, neuron]`. To contract over the shared sample index, that index has to sit on the inside of the product, and transposing `A` is what puts it there. The `.T` is just bookkeeping for "sum over the batch."

## dA: summing over the neurons

Same procedure, different variable. Where does `a1` appear? In `N1` (scaled by `W11`) and in `N2` (scaled by `W12`). It does not appear in `N3` or `N4` at all, because those belong to sample 2.

```
da1 = dN1·W11 + dN2·W12
```

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/neuron-grad/da.svg" alt="Backward diagram: gradients dN1 from neuron 1 and dN2 from neuron 2 both flow back into da1, scaled by W11 and W12, and are summed." style="max-width: 100%; height: auto;"/>
</div>

This time the sum runs over **neurons**, not samples. An input was broadcast out to every neuron in the layer, so every neuron sends something back.

All four:

```
da1 = dN1·W11 + dN2·W12        da2 = dN1·W21 + dN2·W22
da3 = dN3·W11 + dN4·W12        da4 = dN3·W21 + dN4·W22
```

which packs into:

```
dA = dN @ W.T
```

The symmetry with `dW` is worth pausing on, because it is the cleanest way to remember which transpose goes where:

- `dW = A.T @ dN` sums over samples. A weight is shared across the batch.
- `dA = dN @ W.T` sums over neurons. An input is shared across the layer.

Two different kinds of sharing, two different axes to sum over, and the transpose lands wherever it needs to for the shared index to end up on the inside.

## db: the fan-out again

Now the bias, and this is where part 1 comes back.

`b1` is added into `N1` and into `N3`. It is not scaled by anything; the local derivative is `1` in both cases. So:

```
db1 = dN1·1 + dN3·1 = dN1 + dN3
```

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/neuron-grad/db.svg" alt="Two-panel diagram. Forward: b1 fans out to N1 and N3, one arrow per sample. Backward: dN1 and dN3 both return to db1 and are added together." style="max-width: 100%; height: auto;"/>
</div>

The top panel is the forward pass, where one bias fans out to every sample. The bottom panel is the backward pass, where those arrows reverse and meet at the same node. When two arrows arrive at one node, they add. That is the whole mechanism.

Both biases:

```
db1 = dN1 + dN3        db2 = dN2 + dN4
```

which is a column-wise sum, collapsing the batch dimension:

```python
db = dN.sum(axis=0, keepdims=True)     # shape (1, 2)
```

This is [part 1's rule]({% post_url 2026-07-30-broadcast-subtraction-gradient %}) verbatim. Writing `+ b` broadcasts a `(1, 2)` row down to `(2, 2)`, broadcasting is a copy, and copies backprop as sums. The only difference from part 1 is the axis: there the broadcast ran across columns so the sum did too, here it runs down rows so the sum follows.

Note that there is no `.T` anywhere in `db`, and no weight either. The bias is added, not multiplied, so nothing scales its gradient on the way back. It just collects.

## Shape checking

Same sanity net as the earlier posts, and it is especially useful here because it disambiguates the transposes without any calculus.

A gradient has the same shape as the thing it differentiates:

```
A  is (2, 2)  ->  dA  must be (2, 2)
W  is (2, 2)  ->  dW  must be (2, 2)
b  is (1, 2)  ->  db  must be (1, 2)
dN is (2, 2)
```

The 2x2 case hides this a bit since everything is square, so imagine a batch of 32 samples, 784 inputs and 100 neurons:

```
A  (32, 784)      W  (784, 100)      b  (1, 100)      dN  (32, 100)
```

Now only one arrangement works for each:

- `dW` must be `(784, 100)`. From `A (32, 784)` and `dN (32, 100)`, the only product that fits is `A.T @ dN`, which is `(784, 32) @ (32, 100)`.
- `dA` must be `(32, 784)`. The only fit is `dN @ W.T`, which is `(32, 100) @ (100, 784)`.
- `db` must be `(1, 100)`. From `dN (32, 100)`, the 32 has to go, so sum over axis 0.

If you ever forget the formulas, you can rebuild all three from the shapes alone.

## Verifying it

```python
import torch

A = torch.randn(2, 2, requires_grad=True)
W = torch.randn(2, 2, requires_grad=True)
b = torch.randn(1, 2, requires_grad=True)

N = A @ W + b
dN = torch.randn(2, 2)
N.backward(dN)

print(torch.allclose(W.grad, A.T @ dN))                    # True
print(torch.allclose(A.grad, dN @ W.T))                    # True
print(torch.allclose(b.grad, dN.sum(dim=0, keepdim=True))) # True
```

## Where this leaves the series

Three posts, and one idea underneath all of them: **count how many times a variable is used in the forward pass, and the backward pass writes itself.**

- A variable used **once** passes its gradient straight through. That was `A` in the subtraction from part 1.
- A variable used **many times** collects a sum, one term per use. That is the bias here, the broadcast vector in part 1, and every weight in a batched layer.
- A variable used **conditionally** gets gradient only on the paths that were actually taken. That was the max in part 2, where losers got exactly zero.

`dW` and `dA` are just the second case dressed up in matrix notation. The transpose is not extra machinery, it is what a sum over a shared index looks like when you write it as a matrix product.
