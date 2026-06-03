#!/usr/bin/env python3
"""
Tutorial 01 — Foundation: LCEL & Prompt Composition

Author: Cobus Greyling
Purpose: Establish the core mental model of building reliable systems
         using LangChain Expression Language (LCEL).

This is deliberately simple. The goal is not to impress — it is to
internalize the pattern of composing typed, observable, reusable pieces.

Run:
    python tutorial.py

Requirements:
    pip install -r ../../requirements.txt
    export OPENAI_API_KEY=...

You can swap the model provider easily — see the "MODEL CONFIGURATION" section.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env in repo root or current dir
load_dotenv()

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# We use langchain-openai for maximum beginner compatibility.
# Later tutorials show how to use init_chat_model for multi-provider support.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# You can also do this with the unified initializer (recommended in newer code):
# from langchain.chat_models import init_chat_model
# llm = init_chat_model("openai:gpt-4o-mini", temperature=0.7)

llm = ChatOpenAI(
    model="gpt-4o-mini",   # Cheap, fast, excellent for tutorials
    temperature=0.7,
    # max_tokens=500,      # Uncomment to control cost/length
    # timeout=30,
)

# If you want to use Anthropic, Gemini, Grok, or local models, replace the
# llm instantiation above with the appropriate ChatX class and adjust the
# model name. All later tutorials contain commented examples.

print("=== Tutorial 01: LCEL Foundations ===\n")
print(f"Using model: {llm.model_name if hasattr(llm, 'model_name') else 'gpt-4o-mini'}")
print("Provider: OpenAI (change in code to experiment)\n")

# =============================================================================
# PROMPT AS A FIRST-CLASS OBJECT
# =============================================================================
# Never use f-strings for prompts in production systems.
# ChatPromptTemplate gives you:
#   - Role separation (system vs human vs ai)
#   - Variable interpolation with validation
#   - Easy versioning and testing
#   - Partial application

system_prompt = """You are a concise, technically precise assistant who helps
engineers understand AI agent systems.

Answer in at most 3 sentences. Be specific. Avoid marketing language."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Explain {topic} in the context of building reliable AI agents."),
    ]
)

# =============================================================================
# THE CHAIN — LCEL COMPOSITION
# =============================================================================
# The | operator creates a RunnableSequence.
# This is the fundamental building block.

chain = prompt | llm | StrOutputParser()

# We can also insert steps that don't change the main value but add data:
# chain = (
#     prompt
#     | llm
#     | StrOutputParser()
#     | RunnablePassthrough.assign(word_count=lambda x: len(x.split()))
# )

# =============================================================================
# INVOCATION & INSPECTION
# =============================================================================

topic = "tool calling and why it matters more than clever prompting"

print("Topic:", topic)
print("-" * 60)

result = chain.invoke({"topic": topic})

print("Result:\n")
print(result)
print("-" * 60)

# =============================================================================
# STREAMING (critical for UX and for long-running agents)
# =============================================================================

print("\nStreaming version (token by token):\n")

for chunk in chain.stream({"topic": "the difference between chains and agents"}):
    print(chunk, end="", flush=True)
print("\n")

# =============================================================================
# INTROSPECTION — WHAT THE HARNESS ACTUALLY SEES
# =============================================================================
# This is where many beginners stop. Real engineering starts here.

print("\n=== Harness Introspection ===\n")

# 1. See the prompt that will actually be sent
prompt_value = prompt.invoke({"topic": topic})
print("Prompt messages that reach the model:")
for msg in prompt_value.messages:
    print(f"  [{msg.type.upper()}] {msg.content[:120]}...")
print()

# 2. Get the full output with metadata (very useful later)
full_output = (prompt | llm).invoke({"topic": topic})
print("LLM raw response object type:", type(full_output).__name__)
print("Has response_metadata:", bool(getattr(full_output, "response_metadata", None)))

if hasattr(full_output, "response_metadata"):
    meta = full_output.response_metadata
    print("  Finish reason:", meta.get("finish_reason"))
    print("  Model:", meta.get("model_name"))
    # Token usage is often in usage_metadata on newer versions
if hasattr(full_output, "usage_metadata") and full_output.usage_metadata:
    print("  Token usage:", full_output.usage_metadata)

print()

# =============================================================================
# BATCH — RUN THE SAME HARNESS ON MANY INPUTS EFFICIENTLY
# =============================================================================

print("=== Batch example ===\n")

topics = [
    "ReAct agents",
    "evaluation of LLM systems",
    "context window management",
]

batch_results = chain.batch([{"topic": t} for t in topics])

for t, r in zip(topics, batch_results):
    print(f"• {t}:")
    print(f"  {r[:160]}...\n")

# =============================================================================
# HARNESS ENGINEERING TAKEAWAYS (the real content of this tutorial)
# =============================================================================

print("=" * 60)
print("HARNESS ENGINEERING TAKEAWAYS — Tutorial 01")
print("=" * 60)
print("""
1. LCEL turns prompt + model + parser into a single, inspectable, composable unit.
   You can .invoke, .stream, .batch, .ainvoke, and later .bind_tools, .with_config,
   .with_retry, .with_fallbacks, etc. This uniformity is extremely powerful.

2. Prompts are code. Storing them as objects (not strings) lets you:
   - Partially apply variables
   - Test them in isolation
   - Version them
   - Compose them (see later tutorials)

3. The output parser is not cosmetic. It is the first place you enforce
   expectations on model behavior. In Tutorial 02 we make this dramatically stronger.

4. Always keep access to intermediate stages during development.
   Being able to print the exact prompt that reached the model is non-negotiable
   for debugging agents.

5. Streaming is a first-class requirement for any user-facing system.
   Design your harnesses so streaming works from day one.

6. Metadata (token counts, finish reasons, model version) belongs in your
   observability layer. Start capturing it now.

Common anti-patterns this tutorial inoculates against:
- Giant f-string prompts scattered across the codebase
- Treating the LLM call as a black box with no visibility
- Building agents before you can reliably compose simple chains
""")

print("\nNext: Tutorial 02 — Structured Outputs (the contract layer).\n")
print("Run: python ../02-structured-outputs/tutorial.py")
