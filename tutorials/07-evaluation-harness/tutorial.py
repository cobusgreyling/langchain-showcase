#!/usr/bin/env python3
"""
Tutorial 07 — Building an Evaluation Harness

Author: Cobus Greyling

This is the single highest-value tutorial for anyone who wants to ship
real agent systems.

We build a small but complete evaluation harness that you can (and should)
extend for your own projects.

Run:
    python tutorial.py
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

print("=== Tutorial 07: Building an Evaluation Harness ===\n")

# =============================================================================
# THE AGENT UNDER TEST (kept deliberately simple so we can focus on the harness)
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


@tool
def search_kb(query: str) -> str:
    """Search internal knowledge base for company policies and facts."""
    kb = {
        "pto": "25 days per year. 5 days rollover max.",
        "expense": "Over $500 requires pre-approval in finance portal.",
        "pricing": "Pro is $99/user/month billed annually.",
        "support": "Pro: 48h business hours. Enterprise: 8h + 24/7 for P1.",
    }
    q = query.lower()
    for k, v in kb.items():
        if k in q:
            return v
    return "NO_MATCH"


@tool
def web_search(query: str) -> str:
    """Public web search for general knowledge."""
    if "capital of france" in query.lower():
        return "Paris"
    if "current year" in query.lower():
        return "2026"
    return f"[web] {query}"


@tool
def calculate(expr: str) -> str:
    """Basic math."""
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return "ERROR"


agent = create_react_agent(
    model=llm,
    tools=[search_kb, web_search, calculate],
    prompt="You are a precise assistant. Use tools when needed. Be concise and cite sources.",
)

# =============================================================================
# EVAL DATA MODEL
# =============================================================================


@dataclass
class TestCase:
    id: str
    query: str
    category: str  # "internal", "external", "math", "reasoning", "edge"
    expected_tools: Optional[List[str]] = None  # tools that should be called
    must_contain: Optional[List[str]] = None    # substrings that must appear in final answer
    must_not_contain: Optional[List[str]] = None
    notes: str = ""


@dataclass
class EvalResult:
    test_id: str
    query: str
    final_answer: str
    tools_used: List[str]
    latency_ms: float
    passed: bool
    scores: Dict[str, float]  # e.g. {"tool_accuracy": 1.0, "grounded": 0.8, ...}
    failure_reasons: List[str]
    raw_messages: List[Dict[str, Any]]


@dataclass
class EvalReport:
    timestamp: str
    total_cases: int
    passed: int
    pass_rate: float
    avg_latency_ms: float
    results: List[EvalResult]
    summary: str


# =============================================================================
# THE GOLDEN DATASET
# =============================================================================
# This is the most important artifact you will create for any agent project.

TEST_CASES: List[TestCase] = [
    TestCase(
        id="int-001",
        query="How many PTO days do employees get and how many can roll over?",
        category="internal",
        expected_tools=["search_kb"],
        must_contain=["25", "5", "roll"],
        notes="Basic internal policy lookup",
    ),
    TestCase(
        id="int-002",
        query="What is the expense approval threshold and process?",
        category="internal",
        expected_tools=["search_kb"],
        must_contain=["500", "pre-approval", "finance"],
    ),
    TestCase(
        id="ext-001",
        query="What is the capital of France?",
        category="external",
        expected_tools=["web_search"],
        must_contain=["Paris"],
    ),
    TestCase(
        id="math-001",
        query="What is 17 * 4 + 9?",
        category="math",
        expected_tools=["calculate"],
        must_contain=["77"],
    ),
    TestCase(
        id="reason-001",
        query="If Pro costs $99 per month billed annually, what is the effective monthly price?",
        category="reasoning",
        expected_tools=["search_kb", "calculate"],
        must_contain=["99"],
        notes="Requires both lookup and math — good for trajectory testing",
    ),
    TestCase(
        id="edge-001",
        query="Tell me about the company's policy on bringing pets to the office.",
        category="edge",
        expected_tools=None,  # We don't have this info
        must_contain=["don't have", "no information", "not available"],
        notes="Agent should admit lack of knowledge instead of hallucinating",
    ),
]


# =============================================================================
# SCORERS
# =============================================================================


def score_tool_use(result: "RunResult", case: TestCase) -> float:
    """1.0 if all expected tools were used (order ignored for simplicity)."""
    if not case.expected_tools:
        return 1.0
    used = set(result.tools_used)
    required = set(case.expected_tools)
    if required.issubset(used):
        return 1.0
    return 0.0


def score_must_contain(answer: str, case: TestCase) -> float:
    if not case.must_contain:
        return 1.0
    hits = sum(1 for phrase in case.must_contain if phrase.lower() in answer.lower())
    return hits / len(case.must_contain)


def score_must_not_contain(answer: str, case: TestCase) -> float:
    if not case.must_not_contain:
        return 1.0
    bad = sum(1 for phrase in case.must_not_contain if phrase.lower() in answer.lower())
    return 0.0 if bad > 0 else 1.0


def llm_as_judge(query: str, answer: str, case: TestCase) -> float:
    """Cheap LLM judge for semantic correctness and groundedness.

    In real life you would cache these and use a stronger judge model.
    """
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""You are evaluating an AI agent's answer.

Question: {query}

Agent's final answer:
{answer}

Criteria:
- Correctness: Does the answer directly and accurately address the question?
- Groundedness: Does it avoid making up facts not supported by tools or knowledge?
- Conciseness: Is it appropriately brief?

Reply with a single number between 0.0 and 1.0 (two decimals). Only the number."""

    try:
        resp = judge.invoke(prompt)
        score = float(resp.content.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5  # neutral on error


# =============================================================================
# RUNNER
# =============================================================================


@dataclass
class RunResult:
    final_answer: str
    tools_used: List[str]
    latency_ms: float
    messages: List[Any]


def run_agent_on_case(case: TestCase) -> RunResult:
    """Execute the agent and capture trajectory + timing."""
    start = time.time()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": case.query}]},
        config={"configurable": {"thread_id": f"eval-{case.id}"}},
    )
    latency = (time.time() - start) * 1000

    final = result["messages"][-1].content

    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["name"])

    return RunResult(
        final_answer=final,
        tools_used=tools_used,
        latency_ms=latency,
        messages=result["messages"],
    )


def evaluate_case(case: TestCase) -> EvalResult:
    run = run_agent_on_case(case)

    scores: Dict[str, float] = {}
    failure_reasons: List[str] = []

    # Tool use
    tool_score = score_tool_use(run, case)
    scores["tool_accuracy"] = tool_score
    if tool_score < 1.0 and case.expected_tools:
        failure_reasons.append(f"Expected tools {case.expected_tools}, got {run.tools_used}")

    # Lexical checks
    contain_score = score_must_contain(run.final_answer, case)
    scores["must_contain"] = contain_score
    if contain_score < 1.0:
        failure_reasons.append("Missing required phrases")

    not_contain_score = score_must_not_contain(run.final_answer, case)
    scores["must_not_contain"] = not_contain_score
    if not_contain_score < 1.0:
        failure_reasons.append("Contained forbidden phrases")

    # LLM judge (semantic)
    judge_score = llm_as_judge(case.query, run.final_answer, case)
    scores["llm_judge"] = judge_score
    if judge_score < 0.6:
        failure_reasons.append(f"LLM judge gave low score ({judge_score:.2f})")

    # Overall pass rule (tune this for your risk tolerance)
    overall = (
        tool_score * 0.35 +
        contain_score * 0.25 +
        not_contain_score * 0.15 +
        judge_score * 0.25
    )
    passed = overall >= 0.75 and len(failure_reasons) == 0

    # Convert messages to something serializable
    serializable_msgs = []
    for m in run.messages:
        serializable_msgs.append({
            "type": getattr(m, "type", type(m).__name__),
            "content": getattr(m, "content", ""),
            "tool_calls": getattr(m, "tool_calls", None),
        })

    return EvalResult(
        test_id=case.id,
        query=case.query,
        final_answer=run.final_answer,
        tools_used=run.tools_used,
        latency_ms=round(run.latency_ms, 1),
        passed=passed,
        scores=scores,
        failure_reasons=failure_reasons,
        raw_messages=serializable_msgs,
    )


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_report(results: List[EvalResult]) -> EvalReport:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / total if total > 0 else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0

    # Simple summary
    failing = [r for r in results if not r.passed]
    summary_lines = [
        f"Pass rate: {pass_rate:.1%} ({passed}/{total})",
        f"Average latency: {avg_latency:.0f} ms",
    ]
    if failing:
        summary_lines.append(f"Failing cases: {', '.join(r.test_id for r in failing)}")
    else:
        summary_lines.append("All cases passed.")

    return EvalReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        total_cases=total,
        passed=passed,
        pass_rate=round(pass_rate, 3),
        avg_latency_ms=round(avg_latency, 1),
        results=results,
        summary="\n".join(summary_lines),
    )


def print_markdown_report(report: EvalReport):
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(report.summary)
    print()

    for r in report.results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"{status}  {r.test_id}  | {r.query[:55]}...")
        print(f"       Tools: {r.tools_used} | Latency: {r.latency_ms:.0f}ms")
        print(f"       Scores: { {k: round(v, 2) for k, v in r.scores.items()} }")
        if r.failure_reasons:
            print(f"       Reasons: {'; '.join(r.failure_reasons)}")
        print(f"       Answer: {r.final_answer[:160]}{'...' if len(r.final_answer) > 160 else ''}")
        print()


def save_report(report: EvalReport, out_dir: Path = Path("eval_reports")):
    out_dir.mkdir(exist_ok=True)
    ts = report.timestamp.replace(":", "-").replace(".", "-")
    json_path = out_dir / f"eval-report-{ts}.json"
    md_path = out_dir / f"eval-report-{ts}.md"

    # JSON (full fidelity)
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    # Markdown (human friendly)
    lines = [
        f"# Eval Report — {report.timestamp}",
        "",
        report.summary,
        "",
        "## Per-Case Results",
        "",
    ]
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"### {r.test_id} — {status}")
        lines.append(f"**Query:** {r.query}")
        lines.append(f"**Tools used:** {r.tools_used}")
        lines.append(f"**Latency:** {r.latency_ms} ms")
        lines.append(f"**Scores:** {r.scores}")
        if r.failure_reasons:
            lines.append(f"**Failures:** {'; '.join(r.failure_reasons)}")
        lines.append(f"**Answer:**\n\n{r.final_answer}\n")
        lines.append("---")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReports saved to {json_path} and {md_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Running evaluation harness on {len(TEST_CASES)} test cases...\n")

    results: List[EvalResult] = []
    for case in TEST_CASES:
        print(f"Running {case.id}...", end=" ", flush=True)
        res = evaluate_case(case)
        results.append(res)
        status = "PASS" if res.passed else "FAIL"
        print(f"{status} ({res.latency_ms:.0f}ms)")

    report = generate_report(results)
    print_markdown_report(report)
    save_report(report)

    print("\n" + "=" * 70)
    print("HARNESS ENGINEERING TAKEAWAYS — Tutorial 07")
    print("=" * 70)
    print("""
1. The golden dataset (TEST_CASES) is the single most valuable artifact.
   Spend real time curating it. It becomes your definition of "correct."

2. Multiple orthogonal scorers beat any single metric.
   Tool accuracy + lexical checks + LLM judge together are much more robust.

3. The harness must be runnable in < 2 minutes on a laptop during development.
   If running evals is painful, people won't do it.

4. Store the full raw trajectory (raw_messages) for every eval run.
   When a case starts failing you will need to see exactly what the agent did.

5. Treat pass rate as a first-class quality signal alongside latency and cost.
   Add it to your CI pipeline and gate merges on it.

6. This harness is deliberately simple. In real projects you will add:
   - Dataset versioning
   - Caching of LLM-judge results
   - Statistical significance testing
   - Human review queue for borderline cases
   - Comparison reports between two agent versions

You now have the foundation to stop guessing and start measuring.
""")

    print("\nNext: Tutorial 08 — Orchestration & Tool Routing.\n")
    print("Run: python ../08-orchestration/tutorial.py")
