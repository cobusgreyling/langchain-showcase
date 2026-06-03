# Tutorial 08 — Orchestration & Tool Routing

**Author:** Cobus Greyling

## Learning Objectives

- Recognize when a single flat list of tools becomes harmful.
- Implement explicit routing so the model chooses the right capability or sub-agent.
- Demonstrate safe parallel tool execution.
- Build a simple supervisor-style pattern.

## Why This Matters for Harness Engineering

Giving an agent 20+ tools at once is a common source of degraded performance.

The model gets confused, calls the wrong tool, or hallucinates parameters.

Good orchestration is about **reducing the decision space** the model faces at any one step while still giving it access to powerful capabilities when needed.

## What You Will Run

- A router that classifies the user request and dispatches to specialized "experts" (simple functions or sub-agents).
- Parallel tool calling with result merging.
- A tiny supervisor that decides whether to answer directly or delegate.

## How to Run

```bash
python tutorial.py
```

## Exercises

1. Turn the router itself into a structured-output LLM call with a Pydantic `RouteDecision` model.
2. Add a third "expert" (e.g. a small RAG retriever from Tutorial 05) and wire it into the router.
3. Implement a simple map-reduce pattern: break a question into sub-questions, run in parallel, then synthesize.
4. Add latency and cost tracking per expert so the orchestrator can prefer cheaper/faster paths when quality is equivalent.

## Harness Engineering Takeaways

- Tool overload is real. The more tools you expose, the more you need routing, namespacing, or hierarchical agents.
- Routing is itself a reliability technique — it forces an explicit decision before expensive or risky actions.
- Parallelism is powerful but requires deterministic result merging and clear attribution in the final answer.
- Supervisor / multi-agent patterns are a natural evolution once single-agent + router is no longer sufficient.

## Common Pitfalls

- Over-engineering orchestration before you have an evaluation harness (Tutorial 07) to measure whether the complexity actually helps.
- Letting the router LLM have the same full tool list you were trying to avoid.
- Poorly defined expert boundaries ("research expert" that can also do math and email).

## Next

[09 — Guardrails & Resilience Patterns](../09-guardrails-resilience/README.md). We now add the defensive layers that keep agents from causing damage or wasting money.
