#!/usr/bin/env python3
"""
Tutorial 05 — RAG from First Principles

Author: Cobus Greyling

A complete, minimal, but realistic RAG pipeline.

We treat RAG as a harness with distinct stages:
load → split → embed & index → retrieve → pack context → synthesize → ground

Run:
    python tutorial.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("=== Tutorial 05: RAG from First Principles ===\n")

# =============================================================================
# 1. LOAD DOCUMENTS (with metadata — extremely important)
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"

docs = []

for md_file in ["agent_harness_principles.md", "sample_company_faq.md"]:
    path = DATA_DIR / md_file
    if path.exists():
        text = path.read_text(encoding="utf-8")
        # Rich metadata is gold for filtering, citations, and debugging
        doc = Document(
            page_content=text,
            metadata={
                "source": md_file,
                "type": "internal_doc",
                "char_count": len(text),
            },
        )
        docs.append(doc)
        print(f"Loaded: {md_file} ({len(text)} chars)")
    else:
        print(f"WARNING: {path} not found, skipping.")

print(f"\nTotal source documents: {len(docs)}")

# =============================================================================
# 2. CHUNK (the most underrated stage in RAG)
# =============================================================================
# Recursive splitter tries to keep semantic units (paragraphs, sentences) together.

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,          # ~100-150 tokens depending on content
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(docs)

print(f"Produced {len(chunks)} chunks")
print("Example chunk metadata:", chunks[0].metadata if chunks else None)
print("Example chunk text (first 180 chars):")
if chunks:
    print(chunks[0].page_content[:180], "...\n")

# =============================================================================
# 3. EMBED + INDEX (Chroma for simplicity and zero infra)
# =============================================================================
# In production you would use a managed vector DB with proper collection management.

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Use a unique collection per run during development so we don't pollute
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="langchain_showcase_rag_demo",
    # persist_directory="./chroma"  # Uncomment to persist across runs
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},   # Retrieve top 4 chunks — tune this
)

print("Vector store created with", len(chunks), "chunks.\n")

# =============================================================================
# 4. SYNTHESIS PROMPT WITH SOURCE AWARENESS
# =============================================================================

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a careful research assistant.

You must answer ONLY using the provided context documents.
If the context does not contain the answer, say "I don't have sufficient information in the available documents."

Always cite sources inline using the format [source: filename, chunk approx position].

Be concise and quote key phrases when they are definitive.""",
        ),
        (
            "human",
            """Question: {question}

Retrieved context:
{context}

Answer with citations:""",
        ),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# =============================================================================
# 5. BUILD THE RAG CHAIN (LCEL style)
# =============================================================================


def format_docs(docs: list[Document]) -> str:
    """Format retrieved docs with source info for the prompt."""
    formatted = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        formatted.append(f"--- Chunk {i} | Source: {src} ---\n{d.page_content}")
    return "\n\n".join(formatted)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# =============================================================================
# 6. RUN QUERIES + SHOW SOURCES (the part most tutorials skip)
# =============================================================================

questions = [
    "What is the PTO policy and carry-over limit?",
    "What are the recommended first milestones when starting a new agent project?",
    "What is the support SLA for Enterprise customers?",
    "Does the company have a remote work policy?",
]

print("=" * 60)
print("RAG DEMONSTRATION")
print("=" * 60)

for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")

    # Retrieve again so we can show what was actually used (critical for debugging)
    retrieved = retriever.invoke(q)
    print(f"   Retrieved {len(retrieved)} chunks from: {[d.metadata.get('source') for d in retrieved]}")

print("\n" + "=" * 60)

# =============================================================================
# 7. BASIC GROUNDEDNESS CHECK (a cheap but valuable guardrail)
# =============================================================================
# In Tutorial 09/10 we make this more robust.

GROUNDING_PROMPT = ChatPromptTemplate.from_template(
    """Given the following retrieved context and the proposed answer, does the answer
only contain claims that are directly supported by the context?

Answer with a single word: GROUNDED or NOT_GROUNDED

Context:
{context}

Proposed answer:
{answer}
"""
)

grounding_chain = GROUNDING_PROMPT | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

test_q = "What is our current Pro plan pricing?"
test_answer = rag_chain.invoke(test_q)
retrieved_for_test = retriever.invoke(test_q)
context_str = format_docs(retrieved_for_test)

grounded = grounding_chain.invoke({"context": context_str, "answer": test_answer})

print(f"\nGroundedness check for: '{test_q}'")
print(f"Grounded verdict: {grounded.strip()}")
print(f"Answer: {test_answer[:220]}...")

# =============================================================================
# HARNESS ENGINEERING TAKEAWAYS
# =============================================================================

print("\n" + "=" * 60)
print("HARNESS ENGINEERING TAKEAWAYS — Tutorial 05")
print("=" * 60)
print("""
1. Metadata on chunks (source, section, date, permissions) is not optional.
   Without it you cannot debug, filter, or cite.

2. Chunk size and overlap are first-class hyperparameters. Treat them as such.
   Measure their effect on your actual questions.

3. The retriever is a component you can (and should) swap and tune independently
   of the generator. Expose k, search_type, filters, etc. as configuration.

4. Always return sources with the answer. Users (and your eval harness) need them.

5. A groundedness / faithfulness check is one of the highest-value, lowest-cost
   guardrails you can add to a RAG system.

6. RAG is a pipeline, not a single LLM call. Your evaluation must cover retrieval
   quality separately from generation quality (see Tutorial 07).

Production notes:
- Add query classification / routing before retrieval (some questions don't need RAG).
- Consider hybrid search (vector + keyword) early.
- For large corpora, add re-ranking and/or query rewriting.
- Version your index and be able to roll back.
""")

print("\nNext: Tutorial 06 — Stateful Agents & Memory.\n")
print("Run: python ../06-memory-state/tutorial.py")
