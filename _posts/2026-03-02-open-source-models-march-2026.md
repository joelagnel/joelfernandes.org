---
layout: post
title: "Open Source AI Models in March 2026: The Cheapest and Most Capable"
date: 2026-03-02
categories: ai technology
published: false
---

I have been thinking about running local or API-hosted open source models for a while now. The price point has become genuinely attractive, so I decided to zero in on a few contenders and compare them head-to-head. The four models I am looking at are **GLM-5** (Zhipu AI), **Kimi K2.5** (Moonshot AI), **MiniMax-M2.5**, and **Qwen3.5 397B A17B** (Alibaba). All four are Chinese-lab models, which says something about where a lot of the open source momentum is right now.

The short version: GLM-5 is the most well-rounded, Kimi K2.5 is the strongest pure coder, and the speed gap between the two is something you need to factor in. Let me go through each dimension.

---

## Section 1: Intelligence vs. Cost

The first thing I checked was raw intelligence relative to price. [Artificial Analysis](https://artificialanalysis.ai) has a great interactive chart for exactly this -- they compute an "Intelligence Index" across a battery of benchmarks and plot it against token price, so you can see who is in the sweet spot.

[![Intelligence vs Price](/Images/open-source-models-2026/intel-vs-price.png)](https://artificialanalysis.ai/models/glm-5?models=minimax-m2-5%2Ckimi-k2-5%2Cglm-5%2Cqwen3-5-397b-a17b#intelligence-vs-price)
*Intelligence Index vs. input/output token price. Higher and to the left is better. ([Artificial Analysis](https://artificialanalysis.ai))*

GLM-5 and Kimi K2.5 both sit in the upper portion of the chart -- higher intelligence than MiniMax and Qwen at competitive prices. Both land close to the "most attractive quadrant" (high intelligence, low cost).

One thing worth noting: GLM-5 is a bit chatty. It tends to produce more output tokens to arrive at its answers. That drives its per-query cost up somewhat compared to Kimi K2.5, even if the per-token price looks similar. The chart below, which uses Artificial Analysis's "cost to run the Intelligence Index" metric (i.e., total cost to complete their benchmark suite), makes that visible:

[![Intelligence vs Cost to Run](/Images/open-source-models-2026/intel-vs-cost.png)](https://artificialanalysis.ai/models/glm-5?models=minimax-m2-5%2Ckimi-k2-5%2Cglm-5%2Cqwen3-5-397b-a17b&intelligence-index-cost=intelligence-vs-cost#intelligence-vs-cost-to-run-artificial-analysis-intelligence-index)
*Intelligence vs. total cost to run the AI benchmark suite. GLM-5 is slightly more expensive to operate than Kimi K2.5 despite similar raw pricing. ([Artificial Analysis](https://artificialanalysis.ai))*

Kimi K2.5 comes out a bit cheaper in practice once you account for verbosity. So if cost efficiency is the primary driver, Kimi has a small edge. If you want the highest intelligence without worrying too much about token spend, GLM-5 is the pick.

For the exact methodology behind the Intelligence Index, see [Artificial Analysis's documentation](https://artificialanalysis.ai).

---

## Section 2: Can It Survive the Agentic Era?

Raw intelligence scores are one thing. What matters more for real workflows is whether a model can handle multi-step agentic tasks without going off the rails. I came across some benchmark results comparing these four models on exactly that.

First, the agentic benchmarks -- GDPval-AA (real-world agentic task completion) and Terminal-Bench Hard (coding and terminal use under agentic conditions):

![Agentic and Terminal Benchmark Results](/Images/open-source-models-2026/bench-agentic-hallucination.jpg)
*Left: GDPval-AA agentic real-world task scores. Right: Terminal-Bench Hard coding and terminal scores.*

GLM-5 leads on GDPval-AA (46%) with a meaningful gap over Kimi K2.5 (39%). On Terminal-Bench Hard, GLM-5 still leads (43%) but Kimi K2.5 is close behind (41%). Both MiniMax and Qwen trail at 35-36% across both.

Then there is hallucination. For agentic work this is arguably the single most important metric -- a model that confidently makes things up is worse than useless when it is acting autonomously on your behalf.

![Hallucination Rate (1 - Hallucination Rate, higher is better)](/Images/open-source-models-2026/bench-hallucination-rate.jpg)
*1 minus hallucination rate -- higher is better. GLM-5 at 66% is nearly double second place.*

GLM-5 at 66% non-hallucination rate is striking. Kimi K2.5 comes in at 35%, and both MiniMax and Qwen fall off to around 11%. That is not a close race. For any autonomous agent work, GLM-5's factual reliability is a serious advantage.

Where Kimi K2.5 fights back is on pure coding. The SciCode benchmark measures scientific coding ability:

![SciCode Benchmark](/Images/open-source-models-2026/bench-scicode.jpg)
*SciCode coding benchmark. Kimi K2.5 edges ahead of GLM-5 on pure coding tasks.*

Kimi K2.5 takes the top spot here at 49%, with GLM-5 at 46% and the field tighter overall (Qwen and MiniMax at 42-43%). So if your use case is primarily code generation rather than open-ended agentic reasoning, Kimi K2.5 deserves serious consideration.

---

## Section 3: Context Window

Context window matters more than people often acknowledge. Larger windows mean you can feed in more code, longer documents, or more conversation history without truncation.

[![Intelligence vs Context Window](/Images/open-source-models-2026/context-window.png)](https://artificialanalysis.ai/models/glm-5?models=minimax-m2-5%2Ckimi-k2-5%2Cglm-5%2Cqwen3-5-397b-a17b&intelligence-index-cost=intelligence-vs-cost&context-window=intelligence-vs-context-window#intelligence-vs-context-window)
*Intelligence vs. context window size. Kimi K2.5 and Qwen3.5 offer significantly larger windows than GLM-5. ([Artificial Analysis](https://artificialanalysis.ai))*

Kimi K2.5 and Qwen3.5 397B A17B both sit at around 260k tokens, well ahead of GLM-5's roughly 200k. If you are doing long-document analysis or extended code reviews, that gap matters. GLM-5's smaller window is the clearest practical limitation in this comparison.

---

## Section 4: Output Speed

None of this matters if the model is too slow to use interactively. GLM-5 wins here, and the gap with Kimi K2.5 is significant.

[![Output Speed](/Images/open-source-models-2026/output-speed.png)](https://artificialanalysis.ai/models/glm-5?models=minimax-m2-5%2Ckimi-k2-5%2Cglm-5%2Cqwen3-5-397b-a17b&intelligence-index-cost=intelligence-vs-cost&context-window=intelligence-vs-context-window#output-speed)
*Output speed in tokens per second. GLM-5 and Qwen3.5 lead; Kimi K2.5 lags significantly. ([Artificial Analysis](https://artificialanalysis.ai))*

GLM-5 and Qwen3.5 come in around 74-75 tokens per second. MiniMax and Kimi K2.5 drop to around 48-49. For interactive use that difference is noticeable. Kimi K2.5 would still be fine for batch or overnight jobs -- tasks you kick off before bed and check in the morning -- but for real-time interactive work GLM-5 is considerably snappier.

---

## Summary

| Model | Intelligence | Agentic | Hallucination | SciCode | Context | Speed |
|---|---|---|---|---|---|---|
| GLM-5 | Best | Best | Best (66%) | 2nd (46%) | Smaller | Fast (75 t/s) |
| Kimi K2.5 | 2nd | 2nd | 2nd (35%) | Best (49%) | Large | Slow (48 t/s) |
| Qwen3.5 397B | 3rd | 3rd | Poor | 3rd | Large | Fast (74 t/s) |
| MiniMax-M2.5 | 4th | 3rd | Poor | 3rd | Smaller | Mid (49 t/s) |

For most agentic or assistant use cases: **GLM-5**. Best intelligence, best reliability, fastest response. The smaller context window and slightly higher verbosity cost are the tradeoffs.

For coding-heavy workloads where speed is less critical and you want the largest context: **Kimi K2.5**. It beats GLM-5 on SciCode and nearly matches it on terminal coding, with a much bigger window.

I am still evaluating whether to actually run these in production for any of my own workflows. But the price point is attractive enough that I am taking it seriously -- which was not true even six months ago.
