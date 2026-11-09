---
layout: post
title: "Vanilla attention basics and llama KV cache design"
date: 2026-11-09 16:45:00 -0500
categories: [machine-learning]
tags: [attention, llama.cpp, kv-cache, talks]
author: Joel Fernandes
description: "Slides on vanilla attention, KV cache layout, and unified versus split KV caches in llama.cpp."
published: true
---

I recently presented **Vanilla attention basics and llama KV cache design** to a group of colleagues at a paper reading group. Here are the slides from that session.

**[Download the slides (PDF)](/resources/vanilla-attention-basics-and-llama-kv-cache-design.pdf)**

The slides start with queries, keys, and values, work through attention with small numerical examples, then look at how llama.cpp stores and accesses its KV cache. They cover split versus unified caches, attention masking, and the `kv_rows` indexing approach for avoiding work on other sequences' cache entries.

This excerpt ends just before prefix caching. The PDF preserves the step-by-step diagram builds, so there are 95 pages rather than one page per topic.
