---
layout: post
title: "Happy Accidents"
date: 2026-02-24
categories: personal science
draft: false
---

I am often fascinated by happy accidents in history. The most profound, life-changing discoveries of the past were not the result of brilliant master plans -- they were unintentional happy accidents. This article goes over a few examples to inspire the reader.

## Galileo and the Moons of Jupiter

In January 1610, Galileo was not hunting for undiscovered worlds. He had just built a telescope that could magnify objects 20 to 30 times -- far better than anything else available at the time -- and he simply pointed it at the brightest thing in the night sky to see what it looked like up close. Jupiter was the obvious choice. What he saw looked like three tiny stars lined up near the planet, and he did not think much of it at first.

He looked again the next night, expecting Jupiter to have moved past those background stars. Instead, the stars had moved with Jupiter. Over the following weeks he tracked them every night and found a fourth. These were not background stars at all -- they were orbiting Jupiter, just like the Moon orbits Earth. That discovery, born from the simple curiosity of testing a new tool on a bright light, shattered the centuries-old belief that everything in the universe revolves around Earth. [[1]](#bib1)

## The Messy Lab Bench

In 1928, Alexander Fleming was not trying to invent a miracle drug. He was studying staph bacteria, and before heading off on a two-week holiday, he left a pile of bacteria-covered petri dishes sitting out on his lab bench instead of cleaning them up. When he returned on September 3rd, one dish had a problem: a blob of mold had drifted in through an open window and landed on it. A less curious scientist would have thrown it out.

Fleming noticed that around the mold, the bacteria were dead. The mold was secreting something that dissolved them. "That's funny," he reportedly said. That funny observation -- from a dish that should have been cleaned -- became the foundation of antibiotics, medicines that have since saved hundreds of millions of lives. It would take another decade and a world war before Howard Florey and Ernst Chain turned Fleming's accident into a drug people could actually take. [[2]](#bib2)

## The Intelligence Nobody Planned For

For decades, AI researchers had one simple goal: build a program that could guess the next word in a sentence. Feed it enough text from the internet and it would get better at guessing. Nobody expected much beyond a very convincing autocomplete. When researchers scaled up their models from GPT-2 to GPT-3, they were not trying to create a reasoning machine -- they just wanted a better word-predictor.

What came out was something nobody designed for. The model could translate between languages it had never been explicitly taught. It could write working code. It could solve logic puzzles. Researchers call this "emergence" -- abilities that appear spontaneously once a model crosses a certain size. To perfectly predict the next word in a physics paper, it turns out, the model has to learn something about physics. The intelligence was a side effect of doing one simple thing at enormous scale. [[3]](#bib3)

## CRISPR Started with Yogurt

In the 1980s, a Spanish microbiologist named Francisco Mojica kept noticing strange, repeating patterns in bacterial DNA that seemed to serve no purpose. Nobody knew what they were for, so researchers gave them a name -- CRISPR -- and mostly moved on. The mystery sat quietly in the literature for years.

The answer came by accident from food science. Researchers at a dairy company were trying to figure out why the bacteria used to make yogurt kept getting wiped out by viral infections. When they examined those mysterious repeating sequences, they realized the gaps between the repeats were actually fragments of viral DNA -- a kind of biological memory the bacteria kept of past infections. By 2012, Jennifer Doudna and Emmanuelle Charpentier realized they could reprogram this same system to cut any DNA with precision, including human DNA. A tool now being used to treat genetic diseases started because someone wanted to understand why yogurt goes bad. [[4]](#bib4)

---

## My Own Happy Accident

These are my favorite happy accident stories and I'll keep adding to this list. While I am not a full-time researcher at work, a large part of what I do is driven by research -- and these stories inspire me. The most important findings were not the result of someone setting out to find them. They happened because a curious person noticed something odd and did not look away.

One such moment came for me in 2022, while working on the Linux kernel. I was digging through trace data looking at how RCU callbacks were being processed on a mostly-idle system. What I noticed was that a constant low-level trickle of callbacks was confusing the hardware power management -- the system thought it was still busy, even when there was nothing meaningful left to do. Batching those callbacks and flushing them lazily turned out to cut package power draw noticeably on real hardware. I was not looking for a power-saving technique. I was just staring at trace output. That work became what is now known as Lazy RCU in the Linux kernel. [[5]](#bib5)

Happy accident-ing!

---

## Bibliography

<a id="bib1"></a>[1] NASA Science. *Jupiter Moons*. NASA Solar System Exploration. <https://science.nasa.gov/jupiter/jupiter-moons/>

<a id="bib2"></a>[2] Tan, S.Y. and Tatsumura, Y. (2015). *Alexander Fleming (1881-1955): Discoverer of penicillin*. Singapore Medical Journal. National Institutes of Health, PMC. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4520913/>

<a id="bib3"></a>[3] Wei, J. et al. (2022). *Emergent Abilities of Large Language Models*. arXiv:2206.07682. <https://arxiv.org/abs/2206.07682>

<a id="bib4"></a>[4] Broad Institute. *Questions and Answers about CRISPR*. <https://www.broadinstitute.org/what-broad/areas-focus/project-spotlight/questions-and-answers-about-crispr>

<a id="bib5"></a>[5] Fernandes, J. (2022). *Make RCU do less to save power (Lazy RCU)*. Linux Plumbers Conference, 2022. <https://www.joelfernandes.org/resources/lpc2022-rcu-do-less.pdf>
