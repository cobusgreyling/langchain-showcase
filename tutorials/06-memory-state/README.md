# Tutorial 06 — Stateful Agents & Memory

**Author:** Cobus Greyling

## Learning Objectives

- Add short-term memory to agents using LangGraph checkpointers.
- Understand thread-scoped state and how to continue conversations.
- See the difference between raw message history and managed/trimmable history.
- Appreciate the token cost and context risks of unbounded memory.

## Why This Matters for Harness Engineering

Almost every real agent use case is conversational or multi-turn.

Without explicit memory management you get:
- Context explosions
- Leaking state between users
- Inability to resume or branch conversations
- No way to implement "forget this conversation" or summarization

Memory is a first-class architectural concern, not an afterthought.

## What You Will Run

- A stateful ReAct agent using InMemorySaver
- Multiple turns in the same thread
- Inspection of the persisted state
- A demonstration of why you need trimming/summarization strategies

## How to Run

```bash
python tutorial.py
```

## Exercises

1. Replace InMemorySaver with a persistent checkpointer (SQLite or Postgres via langgraph-checkpoint-postgres) and resume across process restarts.
2. Implement a simple history trimmer that keeps only the last N messages + a summary of older turns.
3. Add the ability to "fork" a conversation by using a new thread_id based on the current one.
4. Add a tool that lets the *agent* explicitly request "summarize and compact this thread so far."

## Harness Engineering Takeaways

- Thread ID is the primary key for all state. Get it wrong and users see each other's data.
- Checkpointers give you time-travel and replay for free — incredibly powerful for debugging and evals.
- Unbounded message history is one of the fastest ways to make agents both expensive and unreliable.
- Memory policy (what to keep, when to summarize, what to forget) must be explicit and testable.

## Common Pitfalls

- Using the same thread_id for every user.
- Never trimming history because "the model can handle it" (until it can't).
- Storing sensitive data in the checkpoint without encryption or redaction.

## Next

[07 — Building an Evaluation Harness](../07-evaluation-harness/README.md). This is the most important tutorial in the series for anyone who wants to ship real systems.
