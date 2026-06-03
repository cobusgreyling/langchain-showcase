#!/usr/bin/env python3
"""
Tutorial 04 — ReAct Agents with LangGraph

Author: Cobus Greyling

We now hand the loop management to LangGraph while keeping full visibility.
This is the pattern used in the vast majority of serious LangChain agent work.

Two APIs are shown in comments:
- The classic and extremely widely used: langgraph.prebuilt.create_react_agent
- The newer (2025/2026): langchain.agents.create_agent (with middleware)

We use the prebuilt in the main code for maximum compatibility and documentation.

Run:
    python tutorial.py
"""

from dotenv import load_dotenv
from typing import Annotated

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver   # We'll use this properly in Tutorial 06

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

print("=== Tutorial 04: ReAct Agents with LangGraph ===\n")

# =============================================================================
# TOOLS (reusing the spirit of Tutorial 03, made slightly more realistic)
# =============================================================================


@tool
def search_company_kb(
    query: Annotated[str, "Precise internal search query about policies, products, or past decisions."],
) -> str:
    """Search the company's internal knowledge base.

    Use for anything related to internal processes, pricing, support policies, or historical decisions.
    """
    kb = {
        "vacation": "Full-time employees receive 25 days PTO/year. Up to 5 days roll over. Managers can approve additional unpaid leave.",
        "expense": "All expenses > $500 require pre-approval in the finance system. Submit within 30 days of spend.",
        "pricing": "2026 pricing: Starter $29/mo, Pro $99/mo, Enterprise contact sales. Annual plans receive 16% discount.",
        "support": "Standard support SLA is 48h for Pro, 8h for Enterprise. Critical issues are triaged within 2h 24/7.",
    }
    q = query.lower()
    for key, val in kb.items():
        if key in q:
            return val
    return "NO_MATCH: No internal document matched. Try broadening the query or using web_search."


@tool
def web_search(query: Annotated[str, "Specific public web search query."]) -> str:
    """Search the public internet for current events, technical facts, people, companies, etc."""
    q = query.lower()
    if "current year" in q or "2026" in q:
        return "The current year is 2026."
    if "langgraph" in q:
        return "LangGraph is the graph orchestration framework from LangChain for building stateful, controllable agents."
    if "capital of" in q:
        return "Paris is the capital of France."
    return f"[SIMULATED WEB] Latest relevant result for: {query}"


@tool
def calculate(expression: Annotated[str, "Arithmetic expression using only numbers and + - * / ( )."]) -> str:
    """Perform basic arithmetic. Do not use for business logic or date math."""
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "ERROR: Only basic arithmetic allowed."
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"ERROR: {e}"


@tool
def get_current_user_context() -> str:
    """Returns basic context about the current user (simulated). Use for personalization."""
    return "User role: Engineering Manager. Team size: 7. Preferred communication: concise bullet points + links."


tools = [search_company_kb, web_search, calculate, get_current_user_context]

# =============================================================================
# SYSTEM PROMPT — THE AGENT'S CONSTITUTION
# =============================================================================
# This is as important as the tools.

SYSTEM_PROMPT = """You are a precise, helpful research agent for an engineering organization.

Core rules:
- Use search_company_kb FIRST for any internal company, policy, pricing, or historical question.
- Use web_search only for general or current events knowledge.
- Always show your reasoning briefly before calling tools.
- When you have enough information, give a clear, sourced final answer.
- If information is missing or uncertain, say so explicitly instead of guessing.
- Keep final answers short and scannable (bullets + bold key facts).
"""

# =============================================================================
# CREATE THE AGENT
# =============================================================================
# This single line gives you a full ReAct agent with:
# - Automatic tool binding
# - Message history management
# - Proper state (messages)
# - Built-in handling of tool call → execution → observation

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,           # In newer LangGraph this is often "prompt"
    # checkpointer=...              # Added in Tutorial 06
)

print("Agent created with create_react_agent (LangGraph prebuilt).")
print(f"Tools: {[t.name for t in tools]}\n")

# =============================================================================
# RUN THE AGENT AND INSPECT EVERYTHING
# =============================================================================

question = (
    "Our engineering manager wants to know: What is our current Pro plan pricing for annual billing, "
    "what is the effective monthly cost, and does the support SLA for Pro customers include 24/7 critical issue response? "
    "Also note the current year for context."
)

print("=" * 60)
print("RUNNING AGENT")
print("=" * 60)
print(f"Question: {question}\n")

# The config lets us identify this thread later (crucial for memory & debugging)
config = {"configurable": {"thread_id": "demo-04-react-001"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
)

print("\n" + "=" * 60)
print("FINAL ANSWER FROM AGENT")
print("=" * 60)
final_message = result["messages"][-1]
print(final_message.content)

# =============================================================================
# FULL STATE INSPECTION — THIS IS WHERE REAL ENGINEERING HAPPENS
# =============================================================================

print("\n" + "=" * 60)
print("FULL MESSAGE HISTORY (the agent's actual trajectory)")
print("=" * 60)

for i, msg in enumerate(result["messages"]):
    role = getattr(msg, "type", msg.__class__.__name__)
    if role == "human":
        print(f"\n[USER {i}]")
        print(msg.content)
    elif role in ("ai", "AIMessage"):
        print(f"\n[ASSISTANT {i}]")
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"  TOOL_CALL → {tc['name']}({tc.get('args', {})})")
        if msg.content:
            print(f"  CONTENT: {msg.content[:300]}{'...' if len(msg.content) > 300 else ''}")
    elif role in ("tool", "ToolMessage"):
        print(f"\n[TOOL_RESULT {i}] {getattr(msg, 'name', 'tool')}")
        content = str(msg.content)
        print(f"  {content[:220]}{'...' if len(content) > 220 else ''}")

# =============================================================================
# GRAPH INTROSPECTION
# =============================================================================

print("\n" + "=" * 60)
print("AGENT GRAPH STRUCTURE (for understanding & customization)")
print("=" * 60)

try:
    graph = agent.get_graph()
    print("Nodes:", list(graph.nodes.keys()))
    print("\nMermaid diagram (paste into https://mermaid.live):")
    print(graph.draw_mermaid())
except Exception as e:
    print(f"Could not draw graph: {e}")

# =============================================================================
# HARNESS ENGINEERING TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("HARNESS ENGINEERING TAKEAWAYS — Tutorial 04")
print("=" * 60)
print("""
1. create_react_agent gives you a complete, debuggable ReAct loop in one call.
   The real power comes from what you do *around* it (prompts, tools, state, evals).

2. The entire conversation, including every tool call and observation, lives
   in `result["messages"]`. This is your primary debugging and evaluation surface.

3. System prompt + tool descriptions are the two levers you have for steering
   behavior without changing code. Treat them as configuration that needs testing.

4. Thread IDs (in config) are the key to memory and to "continuing" a previous run.
   This becomes critical in Tutorial 06.

5. Prebuilt agents are excellent starting points. By Tutorial 10 you will be
   building (and testing) your own customized graphs and harness wrappers.

6. Always inspect the *full* trajectory during development. You will be shocked
   at how often the model does something reasonable-looking that is actually wrong.

Next steps in your own work:
- Add a checkpointer immediately (Tutorial 06)
- Add an evaluation harness before you add more tools (Tutorial 07)
- Start logging token usage and latency per step
""")

print("\nNext: Tutorial 05 — RAG from First Principles.\n")
print("Run: python ../05-rag-basics/tutorial.py")
