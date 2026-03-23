---
layout: post
title: "Memory Research in the Age of LLMs"
date: 2026-03-23
published: false
---

*This post is a work in progress. Check back for updates.*

I've been spending a lot of time thinking about how AI agents remember things, and it turns out the answer is: not very well, unless you put serious thought into it. This post will be a deep dive into the full spectrum of memory and search techniques available today, starting from the simplest approaches (full-text search, BM25, keyword matching, and the indexing strategies that make them fast) and working up to embeddings and retrieval-augmented generation (RAG). I'll explore the difference between conversational memory recall (what did we talk about last Tuesday?) and document search (find me that PDF about tax deductions), which turn out to be fundamentally different problems that people keep trying to solve with the same hammer. There's an interesting middle ground on the privacy front: open-source models with cloud-hosted embeddings, which might be the best tradeoff most people aren't considering. I'll then do a hands-on comparison of the tools I've actually used: [qmd](https://github.com/qmdnls/qmd), [txtai](https://github.com/neuml/txtai), [mem0](https://github.com/mem0ai/mem0), and OpenClaw's built-in memory system, covering configuration, reindexing, sync triggers, MCP integration, and hooks. Finally, I'll benchmark the leading open-source embedding models (GTE, E5, BGE, Nomic, Jina) against proprietary ones (OpenAI `text-embedding-3-large`, Google `text-embedding-004`, Cohere `embed-v4`, Voyage) on retrieval quality, dimensionality, latency, and cost, because the number of dimensions your vectors have matters less than you think, and more than the marketing pages admit.

## Outline

### Part 1: Search Techniques, Simple to Complex
- Full-text search (FTS5, Tantivy)
- BM25 scoring and why it's still surprisingly good
- Keyword search and inverted index fundamentals
- Indexing strategies: B-trees, hash indexes, trigram indexes
- When simple search is all you need

### Part 2: Embeddings and RAG
- What embeddings actually capture (and what they miss)
- Chunking strategies and why they matter more than your model choice
- Vector databases vs. bolt-on vector search (pgvector, SQLite-vec, FAISS)
- RAG architectures: naive retrieval vs. reranking vs. hybrid

### Part 3: Conversational Memory vs. Document Search
- Why these are different problems
- Session memory, entity extraction, fact graphs
- The "what did I say last week" problem
- mem0's approach: LLM-extracted facts + graph store

### Part 4: Privacy and the Open Source Question
- Open-source models with cloud embeddings: the pragmatic middle ground
- What actually leaves your machine (and what doesn't)
- Local-only options and their tradeoffs
- Threat models that actually matter

### Part 5: Tool Comparison
- **qmd** — lightweight, Markdown-native
- **txtai** — embeddings + SQL + workflows, Python ecosystem
- **mem0** — conversational memory with LLM extraction, graph backend
- **OpenClaw** — built-in memory search, session memory, mem0 integration
  - Configuration, reindexing triggers, sync hooks, MCP integration
- Comparison matrix: setup complexity, accuracy, latency, privacy, extensibility

### Part 6: Embedding Model Benchmarks
- Open source: GTE-large, E5-mistral-7b, BGE-large, Nomic-embed, Jina-v3, Gemini embedding
- Proprietary: OpenAI text-embedding-3-large, Google text-embedding-004, Cohere embed-v4, Voyage-3
- Dimensions: 384 vs 768 vs 1024 vs 1536 vs 3072 — what actually matters
- MTEB benchmark analysis
- Retrieval quality vs. latency vs. cost per million tokens
- My picks for different use cases
