# Tutorial 02 — Structured Outputs & Validation

**Author:** Cobus Greyling

## Learning Objectives

- Force models to return machine-readable, validated data using Pydantic.
- Understand `with_structured_output` and the two main mechanisms (tool calling vs JSON mode).
- Learn defensive techniques when the model still deviates from the schema.
- Build the habit of defining explicit contracts between agent and application.

## Why This Matters for Harness Engineering

Free text from an LLM is almost never what your downstream code or another agent actually needs.

Structured output turns the model from a "creative writer" into a "typed function."

This is the difference between a demo and a system you can put in a loop, test, and trust.

## What You Will Run

- Basic Pydantic extraction
- Nested models and lists
- Multiple attempts + fallback behavior
- Raw output + parsed output (critical for debugging)
- A small "contract test" pattern you can reuse

## How to Run

```bash
python tutorial.py
```

## Exercises

1. Add a `confidence: float = Field(ge=0, le=1)` field and instruct the model to assess its own certainty. Observe how often it is overconfident.
2. Create a schema for "ActionPlan" with `steps: list[Step]` and `risks: list[str]`. Have the model plan a small research task.
3. Implement a simple retry loop: if parsing fails, feed the error back to the model once with "Fix the previous output to match the schema exactly."
4. Compare `with_structured_output` vs manually asking for JSON + `json.loads` + `model_validate`. Which is more reliable?

## Harness Engineering Takeaways

- Define the data model **before** you write the prompt. The schema is the spec.
- `with_structured_output` is not magic — it is usually implemented by giving the model a tool. The model is still free to ignore it sometimes.
- Always capture the raw generation when using structured output in production. You will need it for debugging and for improving few-shot examples.
- Validation errors are gold. Log them. They tell you where your prompt or model is weak.
- For agents, the output schema of one step often becomes the input contract for the next tool or agent.

## Common Pitfalls

- Making schemas too loose ("just put everything in a big description field").
- Forgetting that some models are much better at structured output than others.
- Not handling the case where the model returns `None` for optional fields that you actually needed.
- Assuming that because it worked in the REPL it will work at 3am with a different model version.

## Next

[03 — Tool Calling Mastery](../03-tool-calling/README.md). Now that the model can speak structured data, we teach it to *use* external capabilities through well-designed tools.
