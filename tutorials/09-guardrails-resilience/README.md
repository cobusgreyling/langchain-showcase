# Tutorial 09 — Guardrails & Resilience Patterns

**Author:** Cobus Greyling

## Learning Objectives

- Implement input and output guardrails as code (not just prompts).
- Add retry logic with exponential backoff for transient failures.
- Build basic cost and token guardrails.
- See how to make an agent fail safely and observably instead of silently.

## Why This Matters for Harness Engineering

Agents will eventually:
- Receive malicious or malformed input
- Generate unsafe or policy-violating output
- Hit rate limits, timeouts, or tool failures
- Consume surprising amounts of money in runaway loops

Guardrails and resilience are what turn a clever demo into a system you can put in front of real users or run unattended.

## What You Will Run

- Input guardrail (length + basic PII-ish patterns)
- Output guardrail (self-critique + length)
- A resilient execution wrapper with retries
- Cost/latency guards that can abort a run

## How to Run

```bash
python tutorial.py
```

## Exercises

1. Add a real PII redaction step (using presidio or simple regex + replacement) before the prompt.
2. Implement a circuit breaker: if the same tool fails 3 times in a row for a thread, stop and escalate.
3. Add a "max total tokens" budget per thread and enforce it across steps.
4. Create a guardrail that detects "the agent is looping" by looking at repeated tool calls or near-identical thoughts.

## Harness Engineering Takeaways

- Guardrails belong in code. Prompts can help, but they are not enforcement.
- Every guardrail trip should be logged with context — this data is pure gold for improving the system.
- Retries must be smart: distinguish transient (rate limit, timeout) from permanent (bad input, auth error).
- Cost control is a guardrail. Unlimited agent runs are a fast way to get a large bill.

## Common Pitfalls

- Relying only on the model to refuse bad requests ("I can't help with that").
- No visibility when a guardrail fires (silent drops or rewrites).
- Infinite retry loops that burn tokens.

## Next

Tutorial 10 — the capstone where we assemble everything into a reusable production harness.
