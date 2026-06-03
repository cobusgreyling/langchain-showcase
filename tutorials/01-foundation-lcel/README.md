# Tutorial 01 — Foundation: LCEL & Prompt Composition

**Author:** Cobus Greyling

## Learning Objectives

- Understand LangChain Expression Language (LCEL) as the fundamental abstraction for building reliable systems.
- Compose `ChatPromptTemplate` + LLM + parser into a clean, inspectable pipeline.
- Appreciate why explicit runnables beat ad-hoc string formatting.
- See the difference between `invoke`, `stream`, and `batch`.

## Why This Matters for Harness Engineering

Every agent you will ever build is ultimately a graph of these same primitives.  
If your mental model is "I send a string to an LLM and get a string back," you will create unmaintainable, untestable, undebuggable systems.

LCEL forces you to think in **composable, observable, typed pipelines**. This is the harness mindset.

## What You Will Run

A simple but complete chain that:
1. Accepts a topic
2. Uses a system + human prompt
3. Calls the model
4. Parses clean output
5. Demonstrates streaming and metadata inspection

## Prerequisites

- Python 3.10+
- OPENAI_API_KEY (or swap the model — see comments in code)

## How to Run

```bash
cd tutorials/01-foundation-lcel
python tutorial.py
```

## Exercises

1. Add a second step in the chain that takes the first answer and asks the model to turn it into a tweet thread (max 3 tweets).
2. Replace `StrOutputParser` with a simple custom parser that extracts a confidence score if the model includes it.
3. Turn the chain into a function that accepts `temperature` as a parameter and returns both the answer and the run metadata.
4. Add `.with_config({"run_name": "foundations-demo"})` and observe the name in LangSmith (if enabled).

## Harness Engineering Takeaways

- **Runnables are the atoms.** Master composition before you reach for agents.
- **Prompts are code.** Treat `ChatPromptTemplate` with the same respect you give to your application code.
- **Streaming is a first-class concern.** If you can't stream, you can't build good UX.
- **Metadata and callbacks are not optional.** You will live in the traces.

## Common Pitfalls

- Building giant f-strings for prompts → impossible to version, test, or reuse.
- Forgetting that `invoke` returns the final value while intermediate steps are lost unless you use `RunnablePassthrough.assign` or similar.
- Assuming the model always returns what the prompt asked for (see Tutorial 02).

## Next

Proceed to [Tutorial 02 — Structured Outputs](../02-structured-outputs/README.md). Structured output is the first major reliability upgrade you can apply to any chain.
