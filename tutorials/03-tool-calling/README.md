# Tutorial 03 — Tool Calling Mastery

**Author:** Cobus Greyling

## Learning Objectives

- Design high-quality tools using the `@tool` decorator and explicit schemas.
- Understand the difference between binding tools and the model deciding to use them.
- Implement a manual tool-calling loop so you deeply understand what "agent" abstractions do under the hood.
- Handle tool errors gracefully so one bad tool call doesn't destroy the whole trajectory.

## Why This Matters for Harness Engineering

Tools are the primary way an agent affects the outside world (and gathers information).

Bad tool design is responsible for the majority of agent failures in production:
- Vague descriptions → model calls the wrong tool or with wrong arguments
- Tools that throw unhelpful exceptions → agent gets stuck in loops
- No distinction between "tool call requested" and "tool call succeeded"

Mastering tools is more important than mastering prompting.

## What You Will Run

- Several well-documented tools
- Manual ReAct-style loop (no magic agent yet)
- Error-resilient tool execution
- Inspection of tool schemas the model actually sees

## How to Run

```bash
python tutorial.py
```

## Exercises

1. Add a tool that can fail in two different ways (transient vs permanent) and have the agent decide whether to retry.
2. Write a tool that returns both a result *and* structured metadata (e.g. "cost", "latency", "source"). Observe how the agent uses the metadata.
3. Create a deliberately bad tool description and watch the model misuse it. Then fix the description and re-run.
4. Implement tool choice forcing (`tool_choice="required"` or specific tool) for cases where you know the model must use a tool.

## Harness Engineering Takeaways

- The docstring + argument annotations of a tool become part of the model's system prompt. Write them like you are writing an API spec for a junior developer.
- Always return useful, structured error information from tools instead of letting exceptions bubble up.
- Separate "the model wants to call X with args Y" from "X succeeded with result Z". Many bugs live in the gap.
- Manual loops are the best way to learn. Only reach for `create_react_agent` after you can write the loop yourself.
- Tool schemas are versioned interfaces. Changing a tool signature is a breaking change for the agent.

## Common Pitfalls

- Tools with 8+ parameters. Models struggle. Prefer narrower tools or a single "config" object.
- Tools that return huge blobs of text without summarization guidance.
- Tools whose names are cute instead of descriptive (`do_the_thing` vs `search_company_registry`).
- Assuming the model will only call tools you intended.

## Next

[04 — ReAct Agents with LangGraph](../04-react-agent/README.md). Now that you understand tools and manual loops, we let the framework manage the loop while we keep full visibility.
