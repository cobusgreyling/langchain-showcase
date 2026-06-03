#!/usr/bin/env python3
"""
Tutorial 02 — Structured Outputs & Validation

Author: Cobus Greyling

This tutorial demonstrates why structured output is the single highest-leverage
reliability technique available when building agents.

We move from "the model wrote some English that looks plausible" to
"the model returned data that matches a contract our code can trust."

Run:
    python tutorial.py
"""

import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

print("=== Tutorial 02: Structured Outputs & Validation ===\n")

# =============================================================================
# 1. DEFINE THE CONTRACT FIRST (this is the engineering step)
# =============================================================================


class AgentCapability(BaseModel):
    """A single capability an AI agent can have."""

    name: str = Field(description="Short, unique capability name, e.g. 'web_search'")
    description: str = Field(description="One-sentence description of what this capability does")
    requires_auth: bool = Field(
        default=False, description="Does this capability require API keys or user credentials?"
    )


class AgentAnalysis(BaseModel):
    """Structured analysis of an agent use case."""

    primary_goal: str = Field(description="The single most important outcome the user wants")
    required_capabilities: List[AgentCapability] = Field(
        description="Capabilities the agent must have (be conservative — only list what is truly necessary)"
    )
    estimated_complexity: int = Field(
        ge=1, le=5, description="1 = simple chain, 5 = multi-agent system with long-term memory and evaluation"
    )
    biggest_risk: str = Field(description="The failure mode most likely to cause user disappointment or system failure")
    recommended_first_milestone: str = Field(
        description="The smallest valuable slice that can be built and evaluated in < 2 days"
    )


# =============================================================================
# 2. THE STRUCTURED LLM
# =============================================================================
# with_structured_output is the recommended high-level API.
# It works by either:
#   a) Using tool calling (the model is given a tool whose schema matches the model)
#   b) JSON mode + parsing (older / some providers)
#
# It returns a validated Pydantic instance (or raises on failure in strict modes).

structured_llm = llm.with_structured_output(AgentAnalysis)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI systems architect who helps teams design reliable agent harnesses.

Be precise and conservative. Prefer fewer, well-scoped capabilities over large vague ones.
Always think about what can go wrong in production.""",
        ),
        ("human", "User request: {request}"),
    ]
)

chain = prompt | structured_llm

print("Example 1: Basic structured extraction\n")

request = "I want an agent that can research competitors, write a short report, and email it to my team every Monday morning."

result: AgentAnalysis = chain.invoke({"request": request})

print("Parsed object type:", type(result).__name__)
print("Primary goal:", result.primary_goal)
print("Complexity:", result.estimated_complexity)
print("Biggest risk:", result.biggest_risk)
print("First milestone:", result.recommended_first_milestone)
print("\nRequired capabilities:")
for cap in result.required_capabilities:
    print(f"  - {cap.name}: {cap.description} (auth={cap.requires_auth})")

print("\n" + "-" * 60 + "\n")

# =============================================================================
# 3. CAPTURING RAW OUTPUT (essential for debugging and iteration)
# =============================================================================
# include_raw=True gives you both the parsed model and the original generation.
# This is gold when the model starts returning garbage.

structured_llm_with_raw = llm.with_structured_output(AgentAnalysis, include_raw=True)

print("Example 2: Structured + raw (for observability)\n")

raw_result = (prompt | structured_llm_with_raw).invoke({"request": request})

print("Parsed model present:", raw_result["parsed"] is not None)
print("Raw content (first 300 chars):")
raw_content = raw_result.get("raw", None)
if raw_content and hasattr(raw_content, "content"):
    print(raw_content.content[:300])
elif isinstance(raw_content, dict):
    print(json.dumps(raw_content, indent=2)[:300])

print("\n" + "-" * 60 + "\n")

# =============================================================================
# 4. NESTED MODELS + LISTS (very common in real harnesses)
# =============================================================================


class Step(BaseModel):
    step_number: int
    action: str = Field(description="What the agent should do in this step")
    success_criteria: str = Field(description="How we will know this step succeeded")


class ResearchPlan(BaseModel):
    question: str
    steps: List[Step] = Field(min_length=1, max_length=6)
    stop_conditions: List[str] = Field(
        description="Conditions under which the agent should stop and report what it has so far"
    )


plan_llm = llm.with_structured_output(ResearchPlan)

plan_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You create tight, testable research plans for AI agents."),
        ("human", "Create a plan for this research question: {question}"),
    ]
)

print("Example 3: Nested plan with list of steps\n")

plan = (plan_prompt | plan_llm).invoke(
    {"question": "What were the main technical reasons the 2024-2025 AI agent frameworks struggled with long-horizon tasks?"}
)

print("Question:", plan.question)
print("Stop conditions:", plan.stop_conditions)
print("\nSteps:")
for s in plan.steps:
    print(f"  {s.step_number}. {s.action}")
    print(f"     Success: {s.success_criteria}")

print("\n" + "-" * 60 + "\n")

# =============================================================================
# 5. DEFENSIVE PATTERNS — WHEN THE MODEL STILL MISBEHAVES
# =============================================================================
# Even with structured output, models can:
# - Return null for required fields
# - Violate constraints
# - Return completely wrong shapes on hard prompts
#
# Good harnesses always have a recovery path.

print("Example 4: Defensive parsing with fallback\n")


class SimpleFact(BaseModel):
    claim: str = Field(description="The core factual claim")
    confidence: float = Field(ge=0.0, le=1.0)
    source_hint: Optional[str] = None


def safe_structured_call(user_input: str, max_retries: int = 1):
    """Production-style helper that never crashes the whole harness."""
    defensive_llm = llm.with_structured_output(SimpleFact, include_raw=True)

    for attempt in range(max_retries + 1):
        try:
            raw = (ChatPromptTemplate.from_template(
                "Extract one crisp fact from the following. Be extremely precise.\n\n{input}"
            ) | defensive_llm).invoke({"input": user_input})

            parsed = raw["parsed"]
            if parsed is None:
                raise ValidationError.from_exception_data("parsed was None", [])

            return parsed, raw["raw"]

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            if attempt == max_retries:
                # Last resort: return a safe default or raise a controlled error
                print("  → Falling back to safe default (harness must continue)")
                return SimpleFact(claim="Unable to extract reliable fact", confidence=0.0), None
    return None, None


fact, raw = safe_structured_call("The capital of France is Paris and it has been since 1789 or something.")
print("Safe fact:", fact.claim, "confidence=", fact.confidence)

# Force a hard case
fact2, _ = safe_structured_call("asdfjkl; no real information here at all just noise 12903812!!")
print("Hard case fact:", fact2.claim, "confidence=", fact2.confidence)

print("\n" + "=" * 60)
print("HARNESS ENGINEERING TAKEAWAYS — Tutorial 02")
print("=" * 60)
print("""
1. The schema is the interface contract. Write it before the prompt.
   Treat Pydantic models as part of your public API between components.

2. with_structured_output dramatically reduces (but does not eliminate)
   output parsing failures. Always keep the raw generation.

3. Constraints (Field(ge=..., le=..., min_length=...)) are free validation.
   Use them aggressively. They turn vague "the model was bad" into
   precise "confidence was 1.3 which violated the schema."

4. Defensive wrappers (safe_structured_call above) belong in every harness.
   Agents run for hours or days. One bad generation must not kill the run.

5. Different models have wildly different structured output quality.
   gpt-4o-mini is surprisingly good. Some open models are terrible.
   Your eval harness (Tutorial 07) must measure this.

6. For agents, the output of one step is often the *input contract* for
   the next tool or sub-agent. Structured output makes that handoff reliable.

Anti-patterns this tutorial helps you avoid:
- "We'll just parse it with regex later"
- Giant free-text "summary" fields that downstream code has to guess at
- No observability into what the model actually emitted before parsing
""")

print("\nNext: Tutorial 03 — Tool Calling Mastery.\n")
print("Run: python ../03-tool-calling/tutorial.py")
