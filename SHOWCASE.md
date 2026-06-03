# LangChain Agent & Harness Engineering — Showcase

> A curated gallery of 10 foundational tutorials designed to take you from basic chains to a complete, production-oriented **agent harness**.

**Author:** Cobus Greyling (sole author)

This page gives you the "why", the learning objectives, the specific harness engineering skills, difficulty, and a teaser for each tutorial. Use it as your map.

> **Prefer a modern visual experience?** Open the live version at https://cobusgreyling.github.io/langchain-showcase/ (or locally `docs/index.html`). It has filtering, search, and nice cards.

---

## How to Use This Showcase

1. Read the one-paragraph **Harness Engineering Focus** for each tutorial.
2. Click into the tutorial folder.
3. Run the `tutorial.py`.
4. Study the **Harness Engineering Takeaways** section (printed + in comments).
5. Do the suggested exercises.

The tutorials are deliberately ordered. Skipping ahead will cost you later.

---

## Tutorial Gallery

### 01 — Foundation: LCEL & Prompt Composition

**Location:** `tutorials/01-foundation-lcel/`

**Time:** 15 min • **Difficulty:** Beginner

**What you will build:** A clean, composable chain using LangChain Expression Language (LCEL) that combines system instructions, user input, and output parsing.

**Harness Engineering Focus:**
- Why treating prompts + LLMs + parsers as first-class composable objects (Runnables) is the foundation of every reliable system.
- How to avoid the "stringly-typed" hell that makes agents unmaintainable.
- The mental model of "pipeline as harness."

**Key concepts:** `ChatPromptTemplate`, `Runnable`, `|` pipe operator, `StrOutputParser`, invoke vs stream.

**Teaser takeaway:** Once you internalize LCEL, every more complex agent is just a bigger graph of the same primitives.

[Open tutorial →](./tutorials/01-foundation-lcel/tutorial.py)

---

### 02 — Structured Outputs & Validation

**Location:** `tutorials/02-structured-outputs/`

**Time:** 20 min • **Difficulty:** Beginner → Intermediate

**What you will build:** Multiple examples of forcing the model to return valid Pydantic models instead of free text.

**Harness Engineering Focus:**
- The single biggest lever for reliability: **make the model speak your schema, not English.**
- How `with_structured_output` works under the hood (tool calling vs JSON mode).
- Validation as a first-class part of the agent loop.
- What to do when the model still hallucinates fields.

**Key concepts:** `BaseModel` + `Field`, `model_config`, `with_structured_output`, `include_raw=True`, fallback strategies.

**Teaser takeaway:** Structured output is not "nice to have." It is the contract between your agent and the rest of your application.

[Open tutorial →](./tutorials/02-structured-outputs/tutorial.py)

---

### 03 — Tool Calling Mastery

**Location:** `tutorials/03-tool-calling/`

**Time:** 25 min • **Difficulty:** Intermediate

**What you will build:** Well-designed tools, proper binding, and a manual tool-calling loop that shows you exactly what the agent runtime does.

**Harness Engineering Focus:**
- Tools are **user interface elements** for the model. Bad descriptions = broken agents.
- How to write tools that fail gracefully and return useful error information.
- The difference between "the model decided to call a tool" and "the tool actually succeeded."
- Why you should understand the manual loop before using the magic agent abstraction.

**Key concepts:** `@tool` decorator, `ToolRuntime`, argument schema inference, manual `tool_calls` handling, error surfacing.

**Teaser takeaway:** 80% of agent failures in production are caused by poorly engineered tools, not bad prompts.

[Open tutorial →](./tutorials/03-tool-calling/tutorial.py)

---

### 04 — ReAct Agents with LangGraph

**Location:** `tutorials/04-react-agent/`

**Time:** 30 min • **Difficulty:** Intermediate

**What you will build:** A real ReAct-style agent using LangGraph's prebuilt agent, with full visibility into the reasoning → action → observation loop.

**Harness Engineering Focus:**
- The agent loop is a **state machine**. Understanding the graph nodes and edges is essential for debugging.
- How to inject custom behavior without forking the framework.
- The importance of clear tool names and descriptions for the ReAct reasoning trace.
- When to use the prebuilt agent vs building your own graph.

**Key concepts:** `create_react_agent`, `StateGraph`, messages state, `graph.get_state()`, interrupts (preview).

**Teaser takeaway:** The ReAct loop is the beating heart of most reliable agents. Master it before you invent new patterns.

[Open tutorial →](./tutorials/04-react-agent/tutorial.py)

---

### 05 — RAG from First Principles

**Location:** `tutorials/05-rag-basics/`

**Time:** 35 min • **Difficulty:** Intermediate

**What you will build:** A complete retrieval-augmented generation pipeline over a small local knowledge base (with a real vector store).

**Harness Engineering Focus:**
- RAG is itself a **harness**: load → transform → index → retrieve → synthesize.
- Chunking strategy and embedding quality are more important than the LLM in most RAG systems.
- How to return sources and build "groundedness" checks.
- Why naive RAG falls apart on real data and what the first mitigations are.

**Key concepts:** `TextLoader` / in-memory docs, `RecursiveCharacterTextSplitter`, embeddings, `Chroma`, retriever, `create_retrieval_chain`, citation handling.

**Teaser takeaway:** Most "RAG doesn't work" problems are actually "we never engineered the retrieval and context packing stage."

[Open tutorial →](./tutorials/05-rag-basics/tutorial.py)

---

### 06 — Stateful Agents & Memory

**Location:** `tutorials/06-memory-state/`

**Time:** 25 min • **Difficulty:** Intermediate

**What you will build:** Agents that remember previous turns using LangGraph checkpointers (short-term memory).

**Harness Engineering Focus:**
- Conversation state is the most dangerous form of hidden global state if not managed explicitly.
- Thread IDs, checkpointing, and how to persist across restarts.
- Trimming history vs summarization (the two real strategies).
- Why "just pass the last 10 messages" is almost never the right engineering answer.

**Key concepts:** `InMemorySaver` (and Postgres/Redis equivalents), `thread_id`, `config`, `get_state`, message trimming.

**Teaser takeaway:** Memory is not a feature you bolt on. It is a core dimension of your harness design.

[Open tutorial →](./tutorials/06-memory-state/tutorial.py)

---

### 07 — Building an Evaluation Harness

**Location:** `tutorials/07-evaluation-harness/`

**Time:** 40 min • **Difficulty:** Intermediate → Advanced (most important tutorial)

**What you will build:** A complete, reusable evaluation harness that runs a battery of tests against your agent and produces a report.

**Harness Engineering Focus:**
- **If you cannot measure it, you cannot improve it.** This is the tutorial that separates demos from systems.
- Golden datasets, deterministic scoring, LLM-as-judge (used responsibly), tool-use verification.
- Regression detection and how to wire this into CI.
- The economics of evaluation (how many tests do you actually need?).

**Key concepts:** Test case schema, `run_eval`, custom scorers, `assert` style for agents, report generation (Markdown + JSON).

**Teaser takeaway:** Your evaluation harness will become the single most valuable piece of infrastructure you own for any agent project.

[Open tutorial →](./tutorials/07-evaluation-harness/tutorial.py)

---

### 08 — Orchestration & Tool Routing

**Location:** `tutorials/08-orchestration/`

**Time:** 30 min • **Difficulty:** Advanced beginner

**What you will build:** An agent that intelligently routes between specialized tools or sub-capabilities, and demonstrates safe parallel tool use.

**Harness Engineering Focus:**
- Single-agent-with-many-tools vs explicit routing vs multi-agent.
- How to prevent "tool overload" (the model gets confused when given 30 tools).
- Parallel execution and result merging without race conditions.
- Designing clean "capability boundaries."

**Key concepts:** Tool choice via LLM router, `RunnableParallel`, conditional edges in graphs, tool namespaces.

**Teaser takeaway:** The art of orchestration is knowing when *not* to give the model every possible tool at once.

[Open tutorial →](./tutorials/08-orchestration/tutorial.py)

---

### 09 — Guardrails & Resilience Patterns

**Location:** `tutorials/09-guardrails-resilience/`

**Time:** 35 min • **Difficulty:** Advanced

**What you will build:** Multiple layers of defense: input validation, output checking, automatic retries with exponential backoff, and basic cost/latency guards.

**Harness Engineering Focus:**
- Guardrails are **code**, not prompts (although prompts can help).
- The difference between "the model refused" and "we programmatically blocked."
- Retry taxonomy: transient vs permanent failures.
- Observability hooks that fire on every guardrail trip.

**Key concepts:** Pre/post processing, custom runnables as middleware, tenacity or manual retry loops, PII redaction patterns, output schema enforcement as guard.

**Teaser takeaway:** Production agents spend more time inside guardrails and retry logic than inside the "happy path" LLM call.

[Open tutorial →](./tutorials/09-guardrails-resilience/tutorial.py)

---

### 10 — The Complete Production Harness

**Location:** `tutorials/10-full-harness/`

**Time:** 45 min • **Difficulty:** Capstone

**What you will build:** A single, reusable `AgentHarness` class (and supporting infrastructure) that combines everything from tutorials 01–09 into a coherent, configurable, observable platform.

**Harness Engineering Focus:**
- Configuration as code (system prompt, tools, model params, memory policy, guardrails).
- Unified execution path with hooks for tracing, metrics, and evaluation.
- How to expose just enough surface area for application developers while hiding LangChain complexity.
- The "harness as product" mindset.

**Key concepts:** Context manager or builder pattern, pluggable components, structured run metadata, cheap "shadow" evals, deployment considerations (what actually matters when you go to prod).

**Teaser takeaway:** By the end you will have a personal/agent-platform skeleton you can copy into real projects instead of starting from `create_react_agent` every time.

[Open tutorial →](./tutorials/10-full-harness/tutorial.py)

---

## Suggested Learning Tracks

**Fast track (engineers who already know LangChain basics):**  
01 (skim) → 02 → 03 → 04 → 07 (critical) → 10

**Deep track (you want to internalize harness thinking):**  
All tutorials in order, plus the exercises in each `README.md`.

**Team curriculum:**  
Use 07 as the forcing function. Require every agent change to come with an updated eval set and a passing harness run.

---

## What Success Looks Like

After completing this showcase you will be able to:

- Explain why a particular agent failed by reading a trace, not by guessing.
- Design tool interfaces that models actually use correctly.
- Stand up an evaluation harness before you write the first line of the "real" agent.
- Make principled decisions about memory, routing, and guardrails instead of cargo-culting examples.
- Package an agent so that other engineers can use it safely without becoming LangChain experts.

---

## Continuing Your Journey

After this showcase, recommended next areas:

- Deep customization of `StateGraph` (custom nodes, human-in-the-loop, persistence)
- Multi-agent systems and supervisor patterns (Deep Agents, Crew patterns, etc.)
- Long-term memory / entity memory architectures
- Production deployment (LangGraph Platform, self-hosted, FastAPI wrappers)
- Advanced evaluation (RAGAS-style metrics, trajectory eval, online evaluation)

---

**Return to main README →** [README.md](./README.md)

---

*This showcase was designed and written by Cobus Greyling to fill the gap between "it works on my machine" demos and systems you can actually run in production with confidence.*
