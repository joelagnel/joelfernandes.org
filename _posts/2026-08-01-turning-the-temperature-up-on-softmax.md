---
layout: post
title: "Turning the Temperature Up on Softmax"
date: 2026-08-01
categories: [ai, llm, machine-learning]
tags: [ai, llm, machine-learning, ml, softmax, temperature, sampling, inference, logits, decoding, greedy-decoding, argmax, probability, neural-networks, hallucination]
author: Joel Fernandes
description: "Softmax barely notices the size of its inputs. What it reacts to is the gaps between them, and temperature is the one knob that stretches or squashes those gaps."
published: true
---

If you have used an LLM API, you have seen a `temperature` parameter, usually somewhere between 0 and 2, usually with a vague note about creativity. Let us see how it works under the hood.

It turns out to be a very small piece of arithmetic, and once you see where it sits, the behavior you get at each end stops being mysterious. We will start with softmax, then look at the one property of softmax that matters here, and then plug temperature into it.

## What softmax does

The last layer of a language model produces a vector of raw scores called logits, one number per token in the vocabulary. They can be negative, they can be large, and they do not add up to anything meaningful. If the vocabulary has 50,000 tokens, you get 50,000 loose numbers.

To sample a token you need a probability distribution instead. Softmax is the function that converts one into the other:

```
p_i = exp(z_i) / sum_j exp(z_j)
```

Read it in two steps. First, exponentiate every logit, which makes all of them positive. Then divide each one by the total, which forces them to sum to 1. That is the whole function, and it gives you the two things a probability distribution needs, for free.

A small example with three tokens:

```python
import numpy as np

z = np.array([2.0, 1.0, 0.0])
p = np.exp(z) / np.exp(z).sum()
print(p)        # [0.665, 0.245, 0.090]
```

The largest logit gets the largest probability, the ordering is preserved, and the three numbers add to 1.

## Softmax barely cares about the values

Here is the part worth internalizing, because everything else follows from it.

Add 100 to every logit and run softmax again:

```python
z = np.array([102.0, 101.0, 100.0])
# [0.665, 0.245, 0.090]     exactly the same as before
```

Nothing moved. Adding a constant `c` to every logit multiplies both the numerator and the denominator by `exp(c)`, and the two cancel. Softmax is shift invariant, so the absolute size of your logits carries no information at all. (This is also why the numerically stable implementation subtracts the row max before exponentiating. It is free.)

Now scale instead of shifting. Double the logits:

```python
z = np.array([4.0, 2.0, 0.0])
# [0.867, 0.117, 0.016]
```

Same ordering, but the winner went from 67% to 87%. The gaps went from 1 apart to 2 apart, and the distribution got noticeably sharper.

So softmax ignores where the logits sit and reacts strongly to how far apart they are. And the reaction is not gentle, because of the exponent. A gap of `d` between two logits becomes a ratio of `exp(d)` between their probabilities:

| logit gap | probability ratio |
|---|---|
| 1 | 2.7x |
| 2 | 7.4x |
| 3 | 20x |
| 5 | 148x |
| 10 | 22,026x |

Differences do not merely survive the trip through softmax. They get amplified, multiplicatively. A logit lead of 3 is not a small edge, it is a 20 to 1 favorite.

## Temperature scales the gaps

Now the knob. Sampling temperature divides the logits by `T` before softmax runs:

```
p_i = exp(z_i / T) / sum_j exp(z_j / T)
```

That is the entire modification. Temperature does not query the model again, does not know anything about the tokens, and cannot change which token the model ranked first. All it does is rescale the gaps, and we just worked out what rescaling gaps does.

- **T below 1** divides by something small, so the gaps grow, the exponent amplifies harder, and the distribution sharpens onto the leading token.
- **T above 1** shrinks the gaps, so the exponent has less to work with and the distribution flattens toward uniform.
- **T equal to 1** leaves the logits alone and gives you the model's own distribution.

The clearest way to see this is with two tokens, where the whole distribution is described by a single number: the gap between the two logits. Plotting the winner's probability against that gap, once per temperature, gives this:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/temp-softmax/temperature-curves.svg" alt="Softmax probability of the leading token plotted against the logit gap, with one colored curve per temperature. Low temperatures form a near vertical step at gap zero, high temperatures form an almost flat line at probability one half." style="max-width: 100%; height: auto;"/>
</div>

Every curve passes through the same point. When the gap is zero the two tokens are tied, and no temperature can break a tie, so the probability is 0.5 regardless. Away from that point the curves separate completely. At `T = 0.1` the curve is nearly a vertical step: a logit gap of half a point is already enough to hand the winner almost all the mass. At `T = 20` the line is nearly flat, and even a gap of 6 only moves the winner to about 57%.

Same model, same logits, same x axis. Only the shape of the response changed.

With a real vocabulary the same thing shows up as the distribution spreading out. Here are five fixed logits at five temperatures:

<div style="margin: 1.5em 0; text-align: center;">
  <img src="/images/temp-softmax/temperature-bars.svg" alt="Five bar charts of the same five logits at temperatures 0.1, 0.5, 1, 2 and 10, showing the probability mass concentrated on one token at low temperature and spread almost evenly at high temperature." style="max-width: 100%; height: auto;"/>
</div>

The top token goes from 99.8% at `T = 0.1`, to 71% at 0.5, to 51% at 1.0, to 36% at 2.0, to 23% at 10. Meanwhile `the`, which the model ranked dead last, climbs from effectively zero to 16%. The model's opinion never changed. Only how much of it we chose to act on.

## The two extremes

**T approaching 0.** Dividing by a number heading toward zero blows every nonzero gap up toward infinity. The largest logit ends up with all of the probability mass and everything else is driven to zero. Softmax degenerates into argmax, and sampling becomes "always take the top token", which is greedy decoding. This is why temperature 0 is described as deterministic. Implementations special case it rather than actually dividing by zero, but the limit is what they are reproducing.

**T approaching infinity.** Now every gap gets crushed toward zero, every exponent tends to `exp(0) = 1`, and each of the N tokens ends up with probability 1/N. The distribution is uniform, and sampling is a uniform random draw over the whole vocabulary. The model still did all its work, and you threw all of it away. The output is noise, because a token the model considered absurd is now just as likely as the one it was confident about.

Neither end is useful on its own, but they bracket the range. Everything you actually use lives in between.

## What this means for creativity

The usual framing is that higher temperature makes a model more creative, and there is something to that, but it is worth being precise about what you are buying.

What higher temperature really does is give the runner up tokens a real chance of being picked. Sometimes that is exactly what you want. In prose or brainstorming, the second and third choices are often perfectly good, and always taking the top one is what makes greedy output feel flat and repetitive. Letting those alternatives through produces phrasing you would not have got otherwise, and it is genuinely useful.

The catch is that the model ranks tokens by likelihood, not by truth, and softmax has no idea which is which. A low probability token can be a fresh turn of phrase or it can be a wrong digit in a date, an invented function name, or a citation that does not exist. Temperature cannot tell those apart, so raising it increases both at once. And because the effect compounds token by token, one unlikely choice early on becomes the context that the rest of the response is conditioned on, so a small wobble can turn into a confidently wrong paragraph.

There is also a practical cost: reproducibility. At `T = 0` the same prompt gives the same output every time, which is what you want when you are debugging, running evaluations, or comparing two prompts. Once you raise the temperature, the same prompt can give different answers on different runs, and a bug that appeared once may not appear again.

Which is why the usual advice holds up. Keep it low for code, extraction, classification, and anything factual, where you want the model's best guess and you want it to be repeatable. Raise it toward 1 or a bit above for writing and idea generation, where variety is worth more than precision, and where a strange word is a much cheaper mistake than a strange fact.

## Wrapping up

Three things worth carrying away:

1. Softmax ignores the absolute size of the logits and responds only to the differences between them.
2. Because of the exponent, those differences are amplified. A gap of `d` becomes a probability ratio of `exp(d)`.
3. Temperature scales the gaps before softmax sees them, which is why one small division can slide the same model all the way from argmax to a uniform random choice.

Nothing about the model changes when you turn the knob. The ranking was fixed the moment the logits came out, and no temperature can reorder it. What you are choosing is how much of the model's own uncertainty you let through into the output.
