#!/usr/bin/env python3
"""
Tutorial 03 — Tool Calling Mastery

Author: Cobus Greyling

This tutorial is deliberately "low level." You will implement the tool-calling
loop yourself before we hand control to LangGraph's agent abstractions.

This understanding is what separates people who can debug agents from
people who can only prompt them.

Run:
    python tutorial.py
"""

import json
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

print("=== Tutorial 03: Tool Calling Mastery ===\n")

# =============================================================================
# DEFINING HIGH-QUALITY TOOLS
# =============================================================================
# The description and parameter docs are CRITICAL.
# The model sees these almost verbatim.


@tool
def search_knowledge_base(
    query: Annotated[str, "Specific, keyword-rich search query. 3-12 words is ideal."],
    top_k: Annotated[int, "Number of results to return. Use 3-5 for precision, up to 10 for recall."] = 5,
) -> str:
    """Search the internal company knowledge base for policies, product info, and past decisions.

    Use this when the question is about company-specific information, not general knowledge.
    Always prefer this over web_search for internal topics.
    """
    # In a real system this would hit a vector DB or API.
    fake_kb = {
        "vacation policy": "Employees receive 25 days of PTO per year. Unused days roll over up to 5 days.",
        "expense approval": "Expenses over $500 require manager approval via the finance portal before submission.",
        "pricing tiers": "Starter: $29/mo, Pro: $99/mo, Enterprise: custom. Annual billing gives 2 months free.",
    }
    q = query.lower()
    hits = [v for k, v in fake_kb.items() if any(word in q for word in k.split())]
    if not hits:
        return "NO_RESULTS: No internal documents matched this query. Consider using web_search instead."
    return "\n".join(hits[:top_k])


@tool
def web_search(
    query: Annotated[str, "Clear, specific search query for general web knowledge."],
) -> str:
    """Search the public web for current events, technical facts, or general knowledge.

    Do NOT use for company-internal information.
    """
    # Simulated. In reality use Tavily, Serper, Bing, etc.
    if "capital of france" in query.lower():
        return "Paris is the capital of France (confirmed 2025)."
    if "current year" in query.lower():
        return "The current year is 2026."
    return f"WEB_RESULT: (simulated) Top result for '{query}' would appear here with title, url, and snippet."


@tool
def calculate(expression: Annotated[str, "A simple mathematical expression using +, -, *, /, and parentheses."]) -> str:
    """Safely evaluate a basic arithmetic expression.

    Only use for pure math. Never use for dates, units, or business logic.
    """
    try:
        # Extremely restricted eval for demo purposes
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "ERROR: Only basic arithmetic is allowed."
        result = eval(expression, {"__builtins__": {}}, {})
        return f"RESULT: {result}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# Register tools
tools = [search_knowledge_base, web_search, calculate]
tools_by_name = {t.name: t for t in tools}

# Bind tools to the model — this is what tells the LLM "you can call these"
llm_with_tools = llm.bind_tools(tools)

print("Tools registered and bound:")
for t in tools:
    print(f"  - {t.name}: {t.description[:70]}...")
print()

# =============================================================================
# INSPECT WHAT THE MODEL ACTUALLY SEES (extremely valuable)
# =============================================================================

print("=== Tool schemas the model receives (simplified) ===\n")
for t in tools:
    schema = t.args_schema.model_json_schema() if t.args_schema else {}
    print(f"Tool: {t.name}")
    print(f"  Description: {t.description}")
    print(f"  Parameters (JSON Schema excerpt): {json.dumps(schema, indent=2)[:400]}...")
    print()

# =============================================================================
# MANUAL TOOL CALLING LOOP — THE HEART OF AGENT ENGINEERING
# =============================================================================
# This is what create_react_agent and create_agent do under the hood.
# Understanding this loop lets you customize, debug, and improve agents.


def run_manual_agent(user_question: str, max_steps: int = 6):
    """A minimal but complete manual agent loop.

    This is the pattern you must understand before trusting higher-level abstractions.
    """
    messages = [HumanMessage(content=user_question)]

    print(f"USER: {user_question}\n")

    for step in range(1, max_steps + 1):
        print(f"--- Step {step} ---")

        # 1. Ask the model what to do (it may call tools or give final answer)
        ai_msg: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # If the model produced a normal text response without tool calls, we're done.
        if not ai_msg.tool_calls:
            print("ASSISTANT (final):", ai_msg.content)
            print()
            return ai_msg.content

        # 2. The model wants to call one or more tools.
        print(f"Model decided to call {len(ai_msg.tool_calls)} tool(s):")
        for tc in ai_msg.tool_calls:
            print(f"  → {tc['name']}({tc['args']})")

        # 3. Execute the tool calls safely.
        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            tool_fn = tools_by_name.get(tool_name)
            if tool_fn is None:
                # Model hallucinated a tool name — this happens.
                result = f"ERROR: No tool named '{tool_name}' exists."
            else:
                try:
                    # Execute the actual tool function
                    result = tool_fn.invoke(tool_args)
                except Exception as e:
                    # This is crucial: tools must never crash the agent loop.
                    result = f"TOOL_EXECUTION_ERROR: {type(e).__name__}: {e}"

            print(f"  Tool '{tool_name}' returned: {str(result)[:160]}...")

            # 4. Feed the observation back as a ToolMessage
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

        print()

    print("Reached max_steps without a final answer. Returning last assistant message.")
    return messages[-1].content if messages else "No response"


# =============================================================================
# DEMOS
# =============================================================================

print("=" * 60)
print("DEMO 1: Question that requires the knowledge base tool")
print("=" * 60)
run_manual_agent("How many PTO days do we get and do they roll over?")

print("=" * 60)
print("DEMO 2: Question requiring calculation + web knowledge")
print("=" * 60)
run_manual_agent("If I buy the Pro plan for a full year, what is my effective monthly cost in 2026?")

print("=" * 60)
print("DEMO 3: Tool that will be misused if description is bad (we use a good one)")
print("=" * 60)
run_manual_agent("What is 15 * (7 + 3) - 40?")

print("=" * 60)
print("DEMO 4: A question where the model should realize it needs external info")
print("=" * 60)
run_manual_agent("Has our company announced any pricing changes in the last 6 months?")

# =============================================================================
# HARNESS ENGINEERING TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("HARNESS ENGINEERING TAKEAWAYS — Tutorial 03")
print("=" * 60)
print("""
1. Tool descriptions and parameter docs are part of the model's prompt.
   Write them with the same care you write production API documentation.

2. The manual loop reveals the two critical state transitions:
   - Model emits tool_calls → we must execute exactly those calls
   - Tool returns observation → we must feed it back with the correct tool_call_id

3. Tool errors are inevitable. A good harness turns tool failures into
   informative observations ("the search returned NO_RESULTS") instead of
   Python exceptions that derail the trajectory.

4. Models will call non-existent tools and will invent argument names.
   Your code must be defensive at the dispatch layer (tools_by_name.get).

5. Parallel tool calling (multiple items in tool_calls) is powerful but
   requires careful result association via tool_call_id. Never assume order.

6. Before you use create_react_agent or create_agent, you should be able
   to write a loop like the one above. The abstractions are convenience,
   not magic.

Production implications:
- Log every tool call + args + result + latency (Tutorial 07 & 10).
- Version your tool schemas. A changed tool is a changed contract for the agent.
- Consider namespacing tools when you have many (search.internal, search.web, ...).
""")

print("\nNext: Tutorial 04 — ReAct Agents with LangGraph (now we let the framework run the loop).\n")
print("Run: python ../04-react-agent/tutorial.py")
