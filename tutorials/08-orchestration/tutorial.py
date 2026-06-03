#!/usr/bin/env python3
"""
Tutorial 08 — Orchestration & Tool Routing

Author: Cobus Greyling

Demonstrates practical orchestration patterns:
- LLM router that chooses a specialized path
- Parallel tool execution with clean merging
- A tiny supervisor that can delegate or answer directly

Run:
    python tutorial.py
"""

import asyncio
from typing import Annotated, Literal
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

print("=== Tutorial 08: Orchestration & Tool Routing ===\n")

# =============================================================================
# SPECIALIZED "EXPERTS" (could be full sub-agents in real life)
# =============================================================================


@tool
def company_policy_lookup(topic: Annotated[str, "Policy area: pto, expense, pricing, support, remote"]) -> str:
    """Look up internal company policy. Only for HR, finance, and support questions."""
    policies = {
        "pto": "25 days/year, 5 rollover max.",
        "expense": ">$500 requires pre-approval via finance portal.",
        "pricing": "Pro $99/user/mo annual. 16% discount vs monthly.",
        "support": "Pro 48h business hours. Enterprise 8h + 24/7 P1.",
        "remote": "Remote-first. Core hours 10-2 local time.",
    }
    return policies.get(topic.lower(), "Policy not found.")


@tool
def web_fact_lookup(query: Annotated[str, "Specific factual question about the world"]) -> str:
    """Answer general knowledge or current events questions using web search (simulated)."""
    q = query.lower()
    if "capital of france" in q:
        return "Paris"
    if "current year" in q or "2026" in q:
        return "2026"
    if "langgraph" in q:
        return "LangGraph is the stateful agent framework from the LangChain team."
    return f"[SIM] Authoritative answer to: {query}"


@tool
def math_calculator(expression: Annotated[str, "Pure arithmetic expression"]) -> str:
    """Perform mathematical calculations. No business logic."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Math error: {e}"


# =============================================================================
# ROUTER (the key orchestration component)
# =============================================================================


class RouteDecision(BaseModel):
    """Structured decision from the router."""

    route: Literal["policy", "facts", "math", "general"] = Field(
        description="Which specialized capability should handle this request."
    )
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(description="Brief explanation for the routing choice")


router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert router for an AI assistant platform.

Available routes:
- policy: internal company HR, finance, support, or process questions
- facts: general world knowledge, current events, technical facts
- math: pure arithmetic calculations
- general: greetings, chit-chat, or questions that don't clearly fit above

Return a structured decision.""",
        ),
        ("human", "{query}"),
    ]
)

router_llm = llm.with_structured_output(RouteDecision)
router = router_prompt | router_llm

print("Router configured.\n")


def route_and_dispatch(query: str) -> str:
    decision: RouteDecision = router.invoke({"query": query})
    print(f"ROUTER → {decision.route} (conf={decision.confidence:.2f}) | {decision.reasoning}")

    if decision.route == "policy":
        # Could call a full sub-agent here
        return company_policy_lookup.invoke({"topic": query})
    elif decision.route == "facts":
        return web_fact_lookup.invoke({"query": query})
    elif decision.route == "math":
        # Extract a clean expression (very naive for demo)
        import re
        expr = re.search(r"[\d\+\-\*\/\(\)\.\s]+", query)
        return math_calculator.invoke({"expression": expr.group(0).strip() if expr else query})
    else:
        return "I'm a specialized assistant. I can help with company policy, facts, or calculations. Please ask a more specific question."


# =============================================================================
# PARALLEL TOOL EXECUTION EXAMPLE
# =============================================================================
# Sometimes you want to call several tools at the same time and merge results.


async def gather_facts_parallel(queries: list[str]) -> list[str]:
    """Run multiple independent lookups concurrently."""
    # In a real system these could be different retrievers or agents.
    tasks = [asyncio.to_thread(web_fact_lookup.invoke, {"query": q}) for q in queries]
    return await asyncio.gather(*tasks)


def parallel_demo():
    print("\n--- Parallel execution demo ---")
    questions = [
        "What is the capital of France?",
        "What year is it?",
        "Tell me about LangGraph",
    ]
    results = asyncio.run(gather_facts_parallel(questions))
    for q, r in zip(questions, results):
        print(f"Q: {q}\n→ {r}\n")


# =============================================================================
# TINY SUPERVISOR PATTERN
# =============================================================================
# The supervisor decides whether it can answer directly or must delegate.

SUPERVISOR_PROMPT = """You are a supervisor for a team of specialized assistants.

You can either:
- ANSWER_DIRECTLY (for simple greetings or when you know the answer with high confidence)
- DELEGATE (to policy, facts, or math specialist)

Reply with exactly one of: ANSWER_DIRECTLY, DELEGATE:policy, DELEGATE:facts, DELEGATE:math
"""

supervisor_chain = (
    ChatPromptTemplate.from_messages([("system", SUPERVISOR_PROMPT), ("human", "{query}")])
    | llm
    | StrOutputParser()
)


def supervisor_route(query: str) -> str:
    decision = supervisor_chain.invoke({"query": query}).strip()
    print(f"SUPERVISOR → {decision}")

    if decision.startswith("DELEGATE:"):
        route = decision.split(":", 1)[1]
        if route == "policy":
            return company_policy_lookup.invoke({"topic": query})
        if route == "facts":
            return web_fact_lookup.invoke({"query": query})
        if route == "math":
            return math_calculator.invoke({"expression": query})
    # default or direct
    return f"Direct answer (supervisor): I can help with policy, facts, and calculations. Your question was: {query}"


# =============================================================================
# DEMOS
# =============================================================================

print("=" * 60)
print("DEMO 1: Router-based orchestration")
print("=" * 60)

queries = [
    "What is our current Pro pricing?",
    "What is the capital of France in 2026?",
    "Calculate 42 * 17 - 100",
    "Tell me a joke about agents",
]

for q in queries:
    print(f"\nUser: {q}")
    answer = route_and_dispatch(q)
    print(f"Final: {answer}")

parallel_demo()

print("\n" + "=" * 60)
print("DEMO 2: Supervisor pattern")
print("=" * 60)

for q in ["Hello!", "How many PTO days do we get?", "What is 9*9?"]:
    print(f"\nUser: {q}")
    print("Answer:", supervisor_route(q))

# =============================================================================
# HARNESS ENGINEERING TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("HARNESS ENGINEERING TAKEAWAYS — Tutorial 08")
print("=" * 60)
print("""
1. Explicit routing reduces the branching factor the model has to consider
   at any moment. This is one of the highest-leverage reliability techniques.

2. Structured output on the router (RouteDecision) makes the decision
   observable, testable, and auditable.

3. Parallel execution is powerful for independent sub-tasks, but you must
   design clean result merging and preserve attribution.

4. Supervisor patterns let you keep a "thin" top level while still having
   deep specialized behavior underneath.

5. Measure the router itself. Track routing accuracy and the cost of
   wrong routes (Tutorial 07 style evals are perfect for this).

6. Start simple (one router + 3-4 experts). Only add hierarchical agents
   or full multi-agent graphs when the eval harness proves you need the
   complexity.

Next: Tutorial 09 — Guardrails & Resilience.
""")

print("\nNext: Tutorial 09 — Guardrails & Resilience Patterns.\n")
print("Run: python ../09-guardrails-resilience/tutorial.py")
