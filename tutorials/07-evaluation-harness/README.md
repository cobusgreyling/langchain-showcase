# Tutorial 07 — Building an Evaluation Harness

**Author:** Cobus Greyling

## Learning Objectives

- Design and implement a reusable evaluation harness for agents.
- Create golden test cases that cover reasoning, tool use, and final answers.
- Implement multiple scoring strategies (heuristic + LLM-as-judge).
- Generate human-readable and machine-readable reports.
- Understand how to run evals as part of development and CI.

## Why This Matters for Harness Engineering

This is the most important tutorial in the entire showcase.

Without an evaluation harness you are flying blind. You cannot reliably improve the agent, you cannot detect regressions when you change the prompt or a tool, and you have no way to compare different models or strategies.

"Works on my prompts" is not a methodology.

## What You Will Run

A complete, self-contained evaluation harness that:
- Defines a small but realistic set of test cases
- Runs your agent (or any callable) against them
- Scores each run on multiple dimensions
- Produces a markdown + JSON report
- Calculates aggregate metrics

You will see both passing and deliberately failing cases.

## How to Run

```bash
python tutorial.py
```

## Exercises (do these — they are the point)

1. Add 3–5 new test cases that cover edge cases you care about (tool misuse, missing info, multi-step reasoning).
2. Improve the LLM judge prompt or add a second judge with different criteria.
3. Add a "tool trajectory" scorer that checks whether specific tools were called in the right order.
4. Wire the harness so it can be called from pytest and fail the test if aggregate score drops below a threshold.
5. Add cost and latency as first-class metrics in the report.

## Harness Engineering Takeaways

- Golden datasets + automated scoring turn agent development from artisanal prompting into engineering.
- LLM-as-judge is useful but noisy. Use it for signals, not as the sole source of truth. Combine with heuristics.
- The act of writing test cases forces you to clarify what "good" actually means for your use case.
- Run evals on every prompt change, tool change, and model upgrade. This is non-negotiable.
- Store eval results with the git commit. You need history of both code *and* quality.

## Common Pitfalls

- Only testing happy paths.
- Using the same model for the agent and the judge (it hides model-specific weaknesses).
- Treating a 70% score as "good enough" without looking at *which* cases fail.
- Not versioning the eval set alongside the code.

## Next

[08 — Orchestration & Tool Routing](../08-orchestration/README.md). Now that you can measure quality, we explore patterns for handling more complex capability sets.
