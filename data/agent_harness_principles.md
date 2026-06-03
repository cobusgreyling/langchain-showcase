# Agent Harness Engineering Principles

## Core Philosophy
An agent harness is the surrounding system that makes an LLM-based agent reliable, observable, testable, and maintainable. Prompt engineering is only a small part of the work.

## Key Principles

1. Contracts over prompts
   Define explicit input and output schemas (Pydantic models) for every major step. The model should speak your types, not free text.

2. Tools are interfaces
   Every tool is a contract with the model. Descriptions, parameter names, and return formats must be written with the same rigor as public APIs.

3. State is explicit
   Never rely on implicit conversation history or global variables. Use checkpointers and thread-scoped state.

4. Evaluation is infrastructure
   You cannot improve what you cannot measure. Every serious agent project needs a regression test suite and automated scoring before it goes to production.

5. Observability first
   Token usage, latency per step, tool success rate, and full trajectories must be captured from day one. LangSmith or equivalent is table stakes.

6. Guardrails are code
   Input validation, output validation, PII redaction, topic enforcement, and cost controls should be implemented in code, not only prompted.

7. Resilience patterns
   Timeouts, retries with backoff, circuit breakers, and graceful degradation are required. Models and tools will fail.

8. Memory management
   Short-term memory (recent messages) must be actively trimmed or summarized. Long-term memory requires careful schema design and retrieval.

## Common Failure Modes

- Context explosion from untrimmed history
- Tool overload (too many tools presented at once)
- Silent failures where the agent guesses instead of admitting ignorance
- Lack of source attribution in RAG leading to ungrounded answers
- No evaluation harness, so regressions go undetected for weeks

## Recommended First Milestones for New Projects

1. One well-scoped tool + structured output + basic evaluation (3-5 test cases)
2. Add memory with a checkpointer
3. Add a second tool and a simple router or ReAct loop
4. Build the evaluation harness to 20-30 cases
5. Add input/output guardrails and retry logic
6. Productionize observability and cost tracking

## RAG Specific Notes

- Chunking strategy often matters more than the embedding model.
- Always return sources with answers.
- "Retrieve then generate" is the minimal harness. Add re-ranking and query rewriting only after you have measurements proving they help.
- Groundedness checking (does the answer only use information from the retrieved chunks?) is a high-value guardrail.
