---
layout: post
title: "Multi-GPU calculation of gradients for accelerated training"
date: 2026-09-09 05:55:00 -0400
categories: [machine-learning]
tags: [gpt-2, pytorch, gradient-accumulation, distributed-training, ddp, cross-entropy]
author: Joel Fernandes
description: "How gradient accumulation makes a half-million-token GPT-2 training batch fit, and how DDP divides that work across GPUs without changing its mean gradient."
math: true
published: true
---

## A half-million-token optimizer step

In [Karpathy's GPT-2 walkthrough](https://www.youtube.com/watch?v=l8pRSuU81PU&t=9313s), he takes the roughly half-million-token batch target from the 125M-parameter row of the [GPT-3 training table](https://arxiv.org/abs/2005.14165) because that model is close in size to GPT-2 124M. In code he chooses the nearby power-of-two value `total_batch_size = 524288`, or `2**19`, tokens per optimizer update. This is a chosen optimizer batch size, not a limit imposed by the GPT-2 architecture.

A **micro-batch** is the data handled by one GPU in one forward and backward call. That call is one **micro-step**. The **global batch** is all data that contributes to one optimizer update, across every micro-step and GPU.

Each sequence contains `T = 1024` tokens, so 524,288 tokens means 512 sequences. Sending all 512 through one GPU at once would require it to retain the intermediate tensor elements, called activations, for all 512 sequences until backward finishes. That does not fit the GPU used in the walkthrough.

Karpathy instead uses `B = 16` sequences per micro-batch:

$$16 \times 1024 = 16{,}384\text{ token predictions}.$$

One GPU therefore needs 32 serial micro-steps to cover the global batch. Each backward pass adds to the parameter gradients, and the optimizer waits until all 32 have finished. Gradient accumulation makes the batch fit by doing the work in smaller pieces. On one GPU, those 32 micro-steps still run sequentially.

## Eight GPUs shorten the serial schedule

The walkthrough starts one process per GPU with `torchrun`, then wraps each process's model in PyTorch DistributedDataParallel, or DDP. Each process has a **rank**. Every rank holds a model replica, and `DataLoaderLite` gives it a different 16-sequence micro-batch.

During one parallel wave, eight ranks process

$$8 \times 16 \times 1024 = 131{,}072\text{ token predictions}$$

at the same time. A **wave** is one micro-step running concurrently on all eight ranks. Four waves cover the global batch:

$$4 \times 131{,}072 = 524{,}288.$$

The one-GPU schedule has 32 serial micro-steps. The eight-GPU schedule has four serial waves. If the computation scaled perfectly, this part would take one eighth as long. In practice, exchanging gradients and other overhead reduce the speedup.

Each rank accumulates gradients from its four micro-batches. On the final backward pass, DDP averages the gradients across all eight ranks. Every model replica then applies the same optimizer update.

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/loss-averaging/gpt2-batch-schedule.png"
       alt="The same 512-sequence global batch scheduled two ways. One GPU processes 16 sequences in each of 32 serial micro-steps. Eight GPUs process eight different 16-sequence chunks at once and need four serial waves before DDP averages their gradients."
       style="max-width: 100%; height: auto;"/>
</div>

*Figure 1. More GPUs reduce the serial work needed for the same global batch. [D2 source](/images/loss-averaging/gpt2-batch-schedule.d2) · [SVG](/images/loss-averaging/gpt2-batch-schedule.svg)*

In the eight-GPU run, `grad_accum_steps = 4`: each rank executes four micro-steps.

```python
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
# 524288 // (16 * 1024 * 8) = 4
```

Each rank therefore divides its losses by four:

```python
loss = loss / grad_accum_steps
```

The factor of eight is absent because DDP supplies the separate average across ranks.

## Cross-entropy averages within one micro-batch

PyTorch receives `B * T` next-token predictions:

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
)
```

The code uses every target token and gives them equal weight. With the default `reduction="mean"`, PyTorch divides the sum of the token losses by `B * T`. Targets from later micro-steps are not arguments to this call, so they are absent from its denominator.

## Gradient accumulation averages across micro-steps

In this GPT-2 loop, every micro-step on every GPU has the same `B * T` valid target tokens. Under that condition, averaging the micro-batch means gives the same global mean as processing all targets together.

Here is a tiny version with two micro-steps and two token predictions per micro-batch. Suppose the individual losses are `2, 4, 6, 8`. Processing all four together would give

$$L_{\text{whole}} = \frac{2+4+6+8}{4}=5.$$

Split them into two micro-batches and cross-entropy returns:

```text
micro-step 1: mean(2, 4) = 3
micro-step 2: mean(6, 8) = 7
```

Calling `.backward()` twice adds into each parameter's `.grad`. Without another division, this gives the gradient of $L_1 + L_2$. We instead want the gradient of their mean:

$$
\nabla\frac{L_1+L_2}{2}
=\nabla\left(\frac{L_1}{2}+\frac{L_2}{2}\right).
$$

For these values:

$$\frac{3}{2}+\frac{7}{2}=5.$$

The identity holds for the loss functions themselves, so their gradients match.

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/loss-averaging/two-microsteps.png"
       alt="An unsplit four-token global batch has mean loss 5. Two micro-steps have cross-entropy means 3 and 7; dividing each by two produces an equivalent mean objective of 5 for the accumulated gradient."
       style="max-width: 100%; height: auto;"/>
</div>

*Figure 2. Cross-entropy supplies the mean inside each micro-batch. Dividing by two supplies the missing mean across two micro-steps. [D2 source](/images/loss-averaging/two-microsteps.d2) · [SVG](/images/loss-averaging/two-microsteps.svg)*

## DDP averages gradients across processes

Now use two GPUs, two micro-steps per rank, and two token predictions per micro-batch.

- GPU 0 sees means `3` and `7`, so its scaled local objective is `3/2 + 7/2 = 5`.
- GPU 1 sees losses `10, 12, 14, 16`. Its micro-batch means are `11` and `15`, so its scaled local objective is `11/2 + 15/2 = 13`.

Call the two local parameter gradients $g_0$ and $g_1$. DDP gives every rank their average, $(g_0 + g_1) / 2$. Using the gradient identity above, that is the gradient of

$$\frac{5+13}{2}=9.$$

This is exactly the whole eight-token mean:

$$\frac{2+4+6+8+10+12+14+16}{8}=9.$$

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/loss-averaging/two-ddp-ranks.png"
       alt="GPU 0 forms local mean objective 5 and GPU 1 forms local mean objective 13. DDP averages the two local parameter gradients, which equals the gradient of the global eight-token mean objective 9."
       style="max-width: 100%; height: auto;"/>
</div>

*Figure 3. Each rank averages across its own micro-steps. DDP performs the final gradient average across ranks. [D2 source](/images/loss-averaging/two-ddp-ranks.d2) · [SVG](/images/loss-averaging/two-ddp-ranks.svg)*

Do not divide the `loss` used for `.backward()` by the number of GPUs. DDP already divides the synchronized parameter gradients by the number of ranks. Dividing by two again in this example would halve the intended gradient.

DDP can skip the gradient all-reduce for the first `grad_accum_steps - 1` micro-steps. The supported `no_sync()` context must wrap both forward and backward. The video instead assigned `model.require_backward_grad_sync` after forward; [a later fix](https://github.com/karpathy/build-nanogpt/commit/f6d194b90f63ee6b7159974078ea60221c172617) moved that assignment before forward because DDP also consults the flag during forward.

## References

- Andrej Karpathy, [gradient accumulation](https://www.youtube.com/watch?v=l8pRSuU81PU&t=9558s) and [DDP gradient averaging](https://www.youtube.com/watch?v=l8pRSuU81PU&t=10839s) in *Let's reproduce GPT-2 (124M)*.
- Tom B. Brown et al., [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165), Table 2 (the 125M-parameter model uses a 0.5M-token batch).
- Karpathy, [video-era cross-entropy call](https://github.com/karpathy/build-nanogpt/blob/ba2554a/train_gpt2.py#L128-L131) and [gradient-accumulation/DDP loop](https://github.com/karpathy/build-nanogpt/blob/ba2554a/train_gpt2.py#L324-L343).
- PyTorch, [`torch.nn.functional.cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy), whose default reduction is `mean`.
- PyTorch, [`DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and the [`no_sync()` gradient-accumulation guidance](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#skip-unnecessary-all-reduce-if-training-with-distributeddataparallel-and-gradient-accumulation).
