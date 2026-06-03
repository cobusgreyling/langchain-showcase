# Tutorial 04 — ReAct Agents with LangGraph

**Author:** Cobus Greyling

## Learning Objectives

- Use `create_react_agent` (or the newer `create_agent`) to get a production-grade ReAct loop with minimal code.
- Inspect the agent's internal state and message history after runs.
- Understand the graph structure so you can customize it later.
- Add simple guardrails and observe the agent's trajectory.

## Why This Matters for Harness Engineering

The ReAct pattern (Reason + Act) is the most reliable and debuggable agent pattern for tool-using agents.

LangGraph gives you:
- Explicit state
- Full history
- The ability to add nodes, edges, and interrupts
- Persistence via checkpointers (see Tutorial 06)

Higher-level "agent" factories are great, but you must still understand the underlying graph.

## What You Will Run

A fully functional research-style agent with 4 tools, asked to solve a multi-step question that requires planning and tool use.

You will print the full message history and graph state.

## How to Run

```bash
python tutorial.py
```

## Exercises

1. Add a `human_in_the_loop` style interrupt before a dangerous tool (simulated "send_email").
2. Print the graph representation (`.get_graph().draw_mermaid()`) and study the nodes.
3. Modify the system prompt to force the agent to always show its reasoning in a specific XML-like format.
4. Give the agent a tool that returns very large output and observe context growth. Then add a summarizer node (preview of advanced patterns).

## Harness Engineering Takeaways

- The agent is a **state machine**, not a black box. Every step is a node transition you can observe and intervene on.
- System prompt + tool descriptions together form the "constitution" of the agent. Change either and behavior changes dramatically.
- Full message history is both a blessing (perfect observability) and a curse (token cost and context limits). Managing it is core harness work.
- Prebuilt agents are starting points, not final products. Plan to customize.

## Common Pitfalls

- Giving the agent 15+ tools without namespacing or routing (see Tutorial 08).
- Weak system prompts that don't tell the agent *when* to stop and answer.
- Not trimming or summarizing history (leads to context explosions).

## Next

[05 — RAG from First Principles](../05-rag-basics/README.md). Retrieval is one of the most important tools an agent can have. We build a proper RAG subsystem.
