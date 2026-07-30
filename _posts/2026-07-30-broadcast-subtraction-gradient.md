---
layout: post
title: "Why the Gradient of a Broadcast Subtraction Is a Sum"
date: 2026-07-30
categories: [ml, backpropagation]
tags: [backpropagation, autograd, broadcasting, gradients, pytorch, neural-networks]
author: Joel Fernandes
description: "A 3x3 matrix minus a 3x1 column vector. The forward pass is trivial. The backward pass quietly sums a whole row, and the reason is worth understanding once, properly."
published: false
---

Here is an operation you write without thinking:

```python
C = A - B     # A is (3, 3), B is (3, 1)
```

The shapes do not match, so broadcasting kicks in and it works anyway. The forward pass is boring. The backward pass is where something slightly surprising happens: the gradient flowing back into `A` is passed through untouched, but the gradient flowing back into `B` is a **sum across each row**, with a minus sign on it.

That summation trips people up the first time they hand-roll backprop. It is not a special rule someone bolted onto the subtraction operator. It falls straight out of the chain rule, and once you see the computational graph, you cannot unsee it.

## The forward pass, written out longhand

Let us name every element instead of using numbers. `A` is our 3x3 matrix:

```
A = | A11  A12  A13 |
    | A21  A22  A23 |
    | A31  A32  A33 |
```

and `B` is a 3x1 column vector:

```
B = | B1 |
    | B2 |
    | B3 |
```

Broadcasting takes that single column and stretches it across all three columns, as if you had written:

```
B_broadcast = | B1  B1  B1 |
              | B2  B2  B2 |
              | B3  B3  B3 |
```

Now the subtraction is elementwise and the result `C` is:

```
C11 = A11 - B1     C12 = A12 - B1     C13 = A13 - B1
C21 = A21 - B2     C22 = A22 - B2     C23 = A23 - B2
C31 = A31 - B3     C32 = A32 - B3     C33 = A33 - B3
```

Read the first row again. `B1` appears three times. Not three different copies of a value, but the *same variable* participating in three different expressions. That single observation is the whole post. Everything below is bookkeeping.

## What we are given, and what we want

Assume `C` feeds into the rest of the network and eventually into a scalar loss `L`. Backprop hands us the upstream gradient:

```
G = dL/dC
```

which has the same shape as `C`, namely 3x3. I will write `G11` for `dL/dC11`, and so on.

Our job at this node is to turn `G` into two things:

- `dL/dA`, which must be 3x3, because `A` is 3x3.
- `dL/dB`, which must be 3x1, because `B` is 3x1.

That shape requirement is already a hint. Something has to collapse nine numbers into three. Hold that thought.

## The A side: the easy half

Take any element, say `A12`. Where does it appear in `C`? In exactly one place: `C12 = A12 - B1`. It touches nothing else.

So the chain rule has only one path to walk:

```
dL/dA12 = dL/dC12 * dC12/dA12
```

And since `C12 = A12 - B1`, the local derivative `dC12/dA12` is just `+1`. Subtracting `B1` does not change how `C12` responds to a wiggle in `A12`: bump `A12` by epsilon, `C12` goes up by epsilon.

```
dL/dA12 = G12 * 1 = G12
```

The same argument holds for all nine elements, so:

```
dL/dA = G
```

The gradient passes through unchanged. No sum, no reshaping, no sign flip. `A` is on the positive side of the minus, and each of its elements has exactly one job in the output.

## The B side: where the sum comes from

Now do `B1`. Ask the same question: where does `B1` appear in `C`?

Three places. `C11`, `C12` and `C13`.

This is a different situation, and it is the situation the multivariable chain rule was invented for. When one variable influences the loss through several distinct paths, the total derivative is the **sum of the contributions along every path**:

```
dL/dB1 = dL/dC11 * dC11/dB1
       + dL/dC12 * dC12/dB1
       + dL/dC13 * dC13/dB1
```

Each local derivative is the same. From `C11 = A11 - B1`, we get `dC11/dB1 = -1`, since `B1` sits after the minus sign: bump `B1` up by epsilon and `C11` goes *down* by epsilon.

Substituting:

```
dL/dB1 = G11 * (-1) + G12 * (-1) + G13 * (-1)
       = -(G11 + G12 + G13)
```

There is the summation. Row 1 of the upstream gradient, added up, negated. The same works out for the other two rows:

```
dL/dB2 = -(G21 + G22 + G23)
dL/dB3 = -(G31 + G32 + G33)
```

Or in one line:

```python
dB = -G.sum(axis=1, keepdims=True)     # shape (3, 1)
```

Two separate things happened here and it is worth keeping them apart in your head:

- The **minus sign** came from `B` being on the subtracted side of the operator.
- The **sum** came from broadcasting, not from the subtraction at all. If the operation had been an addition with the same broadcast, you would still sum; you just would not negate.

## The computational graph

The graph makes this obvious in a way the algebra does not. Below I show just two of the three copies, `C11` and `C12`, because the picture stays readable and the third copy adds nothing new.

Forward, `B1` has two arrows leaving it. It feeds the subtraction that makes `C11`, and it feeds the subtraction that makes `C12`:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/broadcast-grad/forward.svg" alt="Forward computational graph: A11 and B1 feed a minus node producing C11; A12 and the same B1 feed another minus node producing C12; both flow into the loss L." style="max-width: 100%; height: auto;"/>
</div>

That is what broadcasting really is. It is not a resize, it is a **copy**. `B1` was used more than once.

Now reverse every arrow. Gradients flow backward from `L`, through `C11` and `C12`, through the two minus nodes, and out to the inputs:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/broadcast-grad/backward.svg" alt="Backward computational graph: gradients G11 and G12 flow back from the loss, each multiplied by -1 through the minus nodes, and both arrive at B1 where they are added together." style="max-width: 100%; height: auto;"/>
</div>

Look at what arrives at each node:

- `A11` gets exactly one incoming arrow, carrying `+G11`. Nothing to combine.
- `A12` gets exactly one incoming arrow, carrying `+G12`. Nothing to combine.
- `B1` gets **two** incoming arrows, carrying `-G11` and `-G12`. Two arrows landing on the same node means those values have to be combined, and the chain rule says the way you combine them is addition.

So the asymmetry between the two halves of the subtraction is not really about plus and minus. It is about **how many arrows leave each node in the forward pass**.

## The rule worth memorizing

If you only take one thing from this:

> **Fan-out in the forward pass becomes sum-in in the backward pass.**

A node that is used in `k` places sends `k` gradients back, and they add. A copy node, run in reverse, is an add node. That is true for broadcasting, for a weight reused across timesteps in an RNN, for a bias vector added to every row of a batch, for shared weights in a convolution. Same principle every time.

Broadcasting just makes the copy invisible in the source code, which is exactly why the sum in the backward pass looks like it appeared from nowhere.

## Shape checking as a sanity net

There is a cheap trick that catches most gradient bugs without any calculus at all: **a gradient always has the same shape as the thing it is the gradient of.**

- `dL/dA` must be 3x3. `G` is already 3x3. Nothing to do.
- `dL/dB` must be 3x1. `G` is 3x3. Three columns must disappear.

When you need to collapse a broadcast dimension, summing over it is the only operation the chain rule ever gives you. So even if you have forgotten the derivation, the shapes tell you to sum, and the position relative to the minus sign tells you to negate.

This is also exactly what PyTorch's autograd does internally when it un-broadcasts a gradient. It sums over every dimension that was expanded during the forward pass.

## Verifying it

Never trust a hand-derived gradient you have not checked:

```python
import torch

A = torch.randn(3, 3, requires_grad=True)
B = torch.randn(3, 1, requires_grad=True)

C = A - B                 # broadcasting, C is (3, 3)
G = torch.randn(3, 3)     # pretend upstream gradient
C.backward(G)

print(torch.allclose(A.grad, G))                             # True
print(torch.allclose(B.grad, -G.sum(dim=1, keepdim=True)))   # True
```

Both print `True`. The `A` gradient is the upstream gradient verbatim, and the `B` gradient is the negated row sum.

## Wrapping up

The next time a backward pass has a `.sum()` in it that you did not expect, go and look at the forward pass and count how many times the variable was used. If it was used more than once, whether you wrote the copies out explicitly or let broadcasting do it silently, the sum is not an optimization or a convention. It is the chain rule doing the only thing it can do when a variable reaches the loss by more than one route.
