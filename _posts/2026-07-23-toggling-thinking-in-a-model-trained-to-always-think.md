---
layout: post
title: "How Do You Turn Off Thinking In a Model That Was Trained to Always Think?"
date: 2026-07-23
categories: [ai, llm]
tags: [ai, llm, llama-cpp, inference, reasoning, deepseek, autoregressive]
description: "How a runtime flag can switch off a reasoning model's thinking block even when every training example told it to think first, by conditioning the autoregressive prefix rather than changing the weights."
---

Here is a puzzle that sounds like a contradiction. You take a reasoning model, one that was trained on example after example where the assistant dutifully opens a `<think>` block, reasons for a while, closes the block, and only then answers. Every single training example told it: think first. Then someone flips `enable_thinking: false` in llama.cpp, and the reasoning just... stops. The model answers immediately, no visible chain of thought.

If the model was trained to always think, how can a runtime flag talk it out of thinking?

The short version: nobody edited the model. They edited the sentence the model was in the middle of reading.

## The model has no rule that says "always think"

The trick rests on one idea, so it is worth sitting with for a moment. An autoregressive language model does not really learn a rule like "emit a thinking block." It does not carry rules in that sense at all. What it learns is a single conditional probability:

`P(next token | every token that came before it)`

Given the tokens so far, what is the likely next token. It practices this across billions of contexts, and what is remarkable about a large transformer is that this one skill, done well, ends up looking like reasoning, knowledge, and style.

So when the training data looks like this:

```
<Assistant><think>
reasoning reasoning reasoning
</think>
final answer
```

the model is not learning "always think." It is learning two separate continuations:

- After `<Assistant><think>`, the likely next tokens are reasoning.
- After `</think>`, the likely next tokens are the final answer.

Both facts live in the weights at the same time. The model saw reasoning follow the open tag, and it also saw the answer follow the close tag. It has learned both neighborhoods of the sentence.

That second neighborhood is the one we can lead it into.

## Change the prefix, not the network

At inference time, the runtime gets to write the beginning of the sentence before handing control to the model. This is called the generation prefix. Whoever controls the prefix controls which of those two learned continuations the model falls into.

- End the prefix with `<think>` and the model, obediently predicting the next token, starts reasoning.
- End the prefix with `</think>` and the model, just as obediently, behaves as though the reasoning already happened and starts writing the answer.

The model never generated that opening or closing tag. The template stapled it onto the prompt before the model woke up. Same weights, same network, same everything. Only the context changed.

<p style="text-align:center;margin:1.6em 0 0.4em;">
  <img src="/images/thinking-toggle/fork.png" alt="Same weights, two different prefixes, two different behaviors" style="width:100%;max-width:440px;height:auto;" />
</p>

<p style="text-align:center;font-size:0.85em;margin:0 0 1.6em;">Source: <a href="/images/thinking-toggle/fork.d2">fork.d2</a> · vector: <a href="/images/thinking-toggle/fork.svg">fork.svg</a></p>

Notice there is exactly one model in that picture, drawn twice. The "same weights" box on the top path is byte-for-byte identical to the one on the bottom path. The only thing that differs is what we handed it to read.

## What llama.cpp actually does for DeepSeek V4

Concretely, it comes down to a few lines of a Jinja chat template. The current DeepSeek V4 template contains logic that boils down to:

```jinja
{% if thinking %}
  <think>
{% else %}
  </think>
{% endif %}
```

The template maps the runtime's `enable_thinking` argument onto its internal `thinking` variable and picks the tag accordingly:

- Thinking enabled: the prompt ends with `<Assistant><think>`, and the model continues by reasoning.
- Thinking disabled: the prompt ends with `<Assistant></think>`, and the model continues as if the reasoning is already behind it.

Conceptually, the two runs are just:

```
prompt + <think>   =>   reasoning tokens
prompt + </think>  =>   answer tokens
```

The weights are identical in both lines. Only the context presented to the weights changes. The flag is not a switch inside the neural network. It is a decision about how to end the prompt.

## But what if it was truly, only ever trained to think?

Fair challenge. Suppose every single answer in training was wrapped in a reasoning block. No plain answers, ever. Does the trick still work?

Mostly, yes, and here is the subtle reason. Even in that world, during teacher forcing the model still saw final-answer tokens sitting immediately after `</think>`. That transition, close-tag then answer, was in the data every time. So when llama.cpp feeds it `</think>` up front, the model recognizes the neighborhood: "ah, tokens after this marker are supposed to look like a final answer," and it produces one.

What it did not get is the reasoning that normally comes before that marker. It has been dropped into the last act of the play without reading the first two. The consequences are exactly what you would expect:

- Easy questions still come out fine. The answer did not need the scratchpad.
- Hard reasoning questions can get worse, because the working-out that would have carried the answer never happened.
- Some models resist the trick and start reasoning anyway, because the pull toward "think first" is strong enough to override the nudge. It is a little like the colleague who is asked to just give the answer and opens with "sure, but first, some context." You asked it to skip the thinking; it briefly considered that request and reasoned its way to a no.
- If the injected prefix is far outside anything the model saw in training, the output can drift into odd territory.

A model that was explicitly trained on both thinking and non-thinking examples toggles far more reliably, because for that model the marker is a genuine learned mode switch. For a thinking-only model, injecting `</think>` is less a switch and more a stage direction: skip ahead to the answer, please.

## llama.cpp can also physically cut thinking short

Everything above is prompt conditioning: you choose how the sentence starts and let the model flow. There is a second, blunter mechanism that has nothing to do with the prefix, and it is worth not confusing the two.

It is the reasoning budget. The `--reasoning-budget` sampler watches for the thinking start marker and counts the reasoning tokens as they stream out. When the budget runs out, it stops asking the model nicely. It reaches into the sampler and sets every candidate token's logit to negative infinity except the one required to close the block:

```
logit(token) = original   if token is the required </think>
             = -infinity   for every other token
```

Negative infinity means zero probability after softmax. Whatever the model would have preferred next, the only move left on the board is `</think>`. With a budget of zero, this happens on the very first step, so the block closes before any reasoning escapes.

<p style="text-align:center;margin:1.6em 0 0.4em;">
  <img src="/images/thinking-toggle/budget.png" alt="The reasoning-budget sampler forcing the close tag" style="width:100%;max-width:620px;height:auto;" />
</p>

<p style="text-align:center;font-size:0.85em;margin:0 0 1.6em;">Source: <a href="/images/thinking-toggle/budget.d2">budget.d2</a> · vector: <a href="/images/thinking-toggle/budget.svg">budget.svg</a></p>

Worth being precise about what this is: it is not the model deciding to wrap up. It is llama.cpp casting the deciding vote on the model's behalf.

```
Model wants:    another reasoning token
llama.cpp allows: only </think>
```

Useful when reasoning runs long and you need a hard cap. But it is a guillotine, not a gentle nudge, and if it lands mid-thought it can chop off the very reasoning that would have produced a good answer.

## Three controls that are easy to run together

There are three separate knobs here, and they are easy to blur together because they all touch "thinking." They act at completely different layers.

| Control | What it actually does |
|---|---|
| `enable_thinking: false` | Changes the chat-template prefix, usually placing generation after `</think>`. Prompt-level. |
| `--reasoning-budget 0` | Forces an already-open reasoning block to close immediately, by clamping logits. Sampler-level. |
| `--reasoning-format none / deepseek` | Changes how generated reasoning is parsed and returned to you, not whether it was generated. Output-level. |

The first decides where the model starts. The second can cut it off after it starts. The third only decides whether you get to see the reasoning in the response payload. llama.cpp documents these as distinct arguments, and it is easy to reach for one while expecting the behavior of another, which is usually why "turning off thinking" seems to act in three different ways.

## The idea worth keeping

Strip away the templates and samplers and it comes down to a single sentence:

> Thinking is toggled mainly by conditioning the autoregressive sequence, not by switching off part of the neural network.

The transformer runs the same internal computation either way. Every layer fires, every attention head attends, whether thinking is on or off. "Thinking off" does not mean the model computes less. It means the model does not emit a long intermediate reasoning sequence out loud. The scratchpad is hidden, not removed. The machine under the hood is doing exactly as much work as it always does. It has just been asked to keep its notes to itself.
