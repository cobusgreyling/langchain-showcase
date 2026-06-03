# Tutorial 05 — RAG from First Principles

**Author:** Cobus Greyling

## Learning Objectives

- Build a complete retrieval-augmented generation pipeline using real components.
- Understand chunking, embedding, indexing, retrieval, and synthesis as distinct harness stages.
- Return sources and implement basic groundedness.
- Appreciate why naive RAG fails and what the first engineering upgrades are.

## Why This Matters for Harness Engineering

RAG is not "stuff documents into the prompt." It is a multi-stage data pipeline with its own failure modes, evaluation needs, and optimization surface.

Most "our RAG doesn't work" problems are actually retrieval or context-packing problems, not LLM problems.

## What You Will Run

- Load two small Markdown documents
- Split into chunks with sensible metadata
- Create embeddings and a Chroma vector store
- Retrieve relevant chunks for a query
- Synthesize an answer with explicit source citations
- A simple groundedness check

## How to Run

```bash
python tutorial.py
```

This tutorial requires the RAG dependencies (already in root requirements.txt):
- langchain-chroma
- chromadb

## Exercises

1. Experiment with different chunk sizes (200 vs 800 tokens) and overlap. Measure impact on retrieval quality for your questions.
2. Add a small re-ranker (even a simple cross-encoder or LLM rerank of top 8 → top 4).
3. Implement "query rewriting": have the model generate 2-3 alternative queries and retrieve from all of them (multi-query retrieval).
4. Add a "no relevant context" path: if retrieval score is below threshold, the agent should say "I don't have information about that in my knowledge base" instead of hallucinating.

## Harness Engineering Takeaways

- Chunking + metadata strategy is often the highest-ROI change in a RAG system.
- Always surface sources to the user and to downstream consumers. Opaque answers destroy trust.
- Groundedness (answer only uses retrieved content) is a guardrail you can implement with another LLM call or heuristics.
- RAG evaluation is different from agent evaluation: you need retrieval metrics (recall@k, MRR) + generation metrics (groundedness, relevance, faithfulness).

## Common Pitfalls

- Using very large chunks → poor retrieval precision.
- No metadata (source, section, date) → impossible to debug or filter.
- Returning 15 chunks to the LLM → context pollution and lost-in-the-middle problems.
- Treating RAG as a one-time build instead of a continuously evaluated pipeline.

## Next

[06 — Stateful Agents & Memory](../06-memory-state/README.md). Now we add persistent conversation state across turns.
