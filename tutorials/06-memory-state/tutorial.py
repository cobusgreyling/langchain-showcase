#!/usr/bin/env python3
"""
Tutorial 06 — Stateful Agents & Memory

Author: Cobus Greyling

This tutorial shows how to give agents real short-term memory using
LangGraph's checkpointer abstraction.

InMemorySaver is perfect for tutorials and testing.
In production you would use a persistent backend (Postgres, Redis, etc.).

Run:
    python tutorial.py
"""

from dotenv import load_dotenv
from typing import Annotated

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

print("=== Tutorial 06: Stateful Agents & Memory ===\n")

# =============================================================================
# TOOLS
# =============================================================================


@tool
def get_user_preference(key: Annotated[str, "The preference key, e.g. 'name', 'role', 'style'"]) -> str:
    """Look up a stored preference for the current user. Returns the value or 'NOT_FOUND'."""
    # In a real system this would query a database keyed by user_id from context.
    fake_prefs = {
        "name": "Cobus",
        "role": "AI platform engineer",
        "style": "concise with bullet points and code references",
        "favorite_model": "prefers smaller fast models for iteration",
    }
    return fake_prefs.get(key.lower(), "NOT_FOUND")


@tool
def remember_fact(fact: Annotated[str, "A fact the user wants remembered for this conversation."]) -> str:
    """Store a fact for the duration of this thread. In production this would go to long-term memory."""
    # This is a toy. Real memory tools write to a store or the graph state.
    return f"OK, I will remember: {fact}"


tools = [get_user_preference, remember_fact]

# =============================================================================
# THE CHECKPOINTER — THE MEMORY BACKEND
# =============================================================================
# InMemorySaver stores state in a Python dict. Perfect for demos.
# Swap it for a real one without changing agent code.

checkpointer = InMemorySaver()

# =============================================================================
# CREATE STATEFUL AGENT
# =============================================================================

SYSTEM = """You are a helpful personal assistant with memory.

Rules:
- Use the get_user_preference tool when the user refers to personal details.
- Use remember_fact when the user explicitly asks you to remember something.
- Be concise. Reference previous context naturally when relevant.
"""

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM,
    checkpointer=checkpointer,
)

print("Stateful agent created with InMemorySaver checkpointer.\n")

# =============================================================================
# CONVERSATION ACROSS MULTIPLE TURNS (same thread_id)
# =============================================================================

thread_id = "user-cobus-demo-001"
config = {"configurable": {"thread_id": thread_id}}

print("=" * 60)
print(f"THREAD: {thread_id}")
print("=" * 60)

turns = [
    "Hi, my name is Cobus. I'm an AI platform engineer.",
    "What is my role again?",
    "Please remember that I prefer concise answers with code references.",
    "What style do I prefer for answers?",
    "Can you remember that my favorite model family for quick iteration is smaller fast models?",
    "What do you remember about my model preferences?",
]

for i, turn in enumerate(turns, 1):
    print(f"\n[TURN {i}] USER: {turn}")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": turn}]},
        config=config,
    )
    last = result["messages"][-1]
    print(f"[TURN {i}] ASSISTANT: {last.content}")

# =============================================================================
# INSPECT THE PERSISTED STATE (this is incredibly powerful)
# =============================================================================

print("\n" + "=" * 60)
print("INSPECTING PERSISTED STATE (what the checkpointer actually saved)")
print("=" * 60)

state = agent.get_state(config)
print(f"Number of messages in checkpoint: {len(state.values.get('messages', []))}")

print("\nLast few messages (role + short content):")
for msg in state.values.get("messages", [])[-6:]:
    role = getattr(msg, "type", type(msg).__name__)
    content = getattr(msg, "content", "")[:100]
    print(f"  {role}: {content}...")

print("\nCheckpoint metadata keys:", list(state.metadata.keys()) if state.metadata else "none")
print("Next checkpoint pointer exists:", bool(state.next))

# =============================================================================
# DIFFERENT THREAD = CLEAN SLATE (isolation)
# =============================================================================

print("\n" + "=" * 60)
print("DIFFERENT THREAD (state isolation demo)")
print("=" * 60)

other_config = {"configurable": {"thread_id": "other-user-999"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi, what is my name?"}]},
    config=other_config,
)
print("Other thread answer:", result["messages"][-1].content)
print("(Correctly does not know — different thread)")

# =============================================================================
# HARNESS ENGINEERING TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("HARNESS ENGINEERING TAKEAWAYS — Tutorial 06")
print("=" * 60)
print("""
1. The checkpointer + thread_id combination gives you:
   - Automatic memory across turns
   - Time travel / replay for debugging
   - Natural branching (just use a new thread_id)

2. InMemorySaver is only for development and tests.
   Production requires a durable backend. The agent code does not change.

3. Message history grows linearly. You *will* hit context limits.
   Every serious harness needs an explicit memory policy:
   - Keep last N messages
   - Summarize older turns (with another LLM call)
   - Or both (recent messages + summary)

4. Thread IDs are security boundaries. Treat them like session tokens.

5. The state you see in get_state is the *source of truth* for what the
   agent "knows." Your eval harness and UI should be able to read it.

6. Long-term memory (across threads/users) is a different architectural layer
   (vector store of facts, entity memory, etc.). Short-term (this tutorial)
   is about conversation continuity within one interaction.

Next up: Tutorial 07. If you only do one tutorial deeply, make it the evaluation harness.
""")

print("\nNext: Tutorial 07 — Building an Evaluation Harness.\n")
print("Run: python ../07-evaluation-harness/tutorial.py")
