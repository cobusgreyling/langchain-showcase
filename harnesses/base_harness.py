"""
Reusable base harness for LangChain agents.

This is an intentionally clean, opinionated starting point extracted from
the capstone of the LangChain Agent & Harness Engineering Showcase.

Author: Cobus Greyling (2026)

Use this as a foundation, not as dogma. Adapt ruthlessly to your domain.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class RunMetadata:
    run_id: str
    thread_id: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"
    latency_ms: float = 0.0
    tokens_estimate: int = 0
    model: str = ""
    tools_used: List[str] = field(default_factory=list)
    guardrail_trips: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class HarnessConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens_per_run: int = 12000
    max_retries: int = 2
    enable_input_guardrails: bool = True
    enable_output_guardrails: bool = True
    system_prompt: str = "You are a precise, helpful agent. Use tools when needed."


def _default_input_guardrail(text: str) -> tuple[bool, str, str]:
    if len(text) > 3500:
        return False, "", "INPUT_TOO_LONG"
    if "ignore all previous instructions" in text.lower():
        return False, "", "JAILBREAK_ATTEMPT"
    return True, text, ""


def _default_output_guardrail(text: str) -> tuple[bool, str, str]:
    if len(text) > 2500:
        return False, "", "OUTPUT_TOO_VERBOSE"
    return True, text, ""


class AgentHarness:
    """
    Production-leaning wrapper around a LangGraph agent.

    Responsibilities:
    - Own configuration and instantiation of the underlying agent
    - Enforce guardrails and budgets
    - Return rich, structured metadata on every run
    - Provide a stable surface for adding tracing, evaluation, etc.

    This class is deliberately small. Add what your system needs.
    """

    def __init__(
        self,
        config: HarnessConfig,
        tools: List[Callable],
        input_guardrail: Optional[Callable[[str], tuple[bool, str, str]]] = None,
        output_guardrail: Optional[Callable[[str], tuple[bool, str, str]]] = None,
    ):
        self.config = config
        self.tools = tools
        self._input_guard = input_guardrail or _default_input_guardrail
        self._output_guard = output_guardrail or _default_output_guardrail

        self.llm = ChatOpenAI(model=config.model_name, temperature=config.temperature)
        self.checkpointer = InMemorySaver()

        self.agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=config.system_prompt,
            checkpointer=self.checkpointer,
        )

        self._budget_used = 0

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 3)

    def invoke(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        run_id = f"run_{self._new_id()}"
        thread_id = thread_id or f"thread_{self._new_id()}"
        start = time.time()

        meta = RunMetadata(
            run_id=run_id,
            thread_id=thread_id,
            start_time=datetime.utcnow().isoformat() + "Z",
            model=self.config.model_name,
        )

        # Input guard
        if self.config.enable_input_guardrails:
            ok, sanitized, reason = self._input_guard(user_input)
            if not ok:
                meta.status = "BLOCKED"
                meta.guardrail_trips.append(f"input:{reason}")
                meta.end_time = datetime.utcnow().isoformat() + "Z"
                return {"answer": "Blocked by input guardrail.", "metadata": asdict(meta)}

            user_input = sanitized

        if self._budget_used > self.config.max_tokens_per_run:
            meta.status = "BUDGET_EXCEEDED"
            meta.end_time = datetime.utcnow().isoformat() + "Z"
            return {"answer": "Budget exceeded.", "metadata": asdict(meta)}

        # Execute with limited retries
        last_err = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                result = self.agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config={"configurable": {"thread_id": thread_id}},
                )

                final_msg = result["messages"][-1]
                answer = getattr(final_msg, "content", str(final_msg))

                tools_used: List[str] = []
                for m in result["messages"]:
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        tools_used.extend(tc.get("name", "") for tc in m.tool_calls)
                meta.tools_used = [t for t in tools_used if t]

                # Output guard
                if self.config.enable_output_guardrails:
                    safe, final_answer, out_reason = self._output_guard(answer)
                    if not safe:
                        meta.status = "BLOCKED"
                        meta.guardrail_trips.append(f"output:{out_reason}")
                        answer = "Response blocked by output guardrail."
                    else:
                        answer = final_answer

                meta.status = "OK"
                meta.latency_ms = round((time.time() - start) * 1000, 1)
                meta.tokens_estimate = self._estimate_tokens(user_input) + self._estimate_tokens(answer)
                self._budget_used += meta.tokens_estimate
                meta.end_time = datetime.utcnow().isoformat() + "Z"

                return {
                    "answer": answer,
                    "metadata": asdict(meta),
                    "thread_id": thread_id,
                    "run_id": run_id,
                }

            except Exception as e:
                last_err = e
                transient = "rate" in str(e).lower() or "timeout" in str(e).lower()
                if attempt > self.config.max_retries or not transient:
                    break
                time.sleep(0.5 * attempt)

        meta.status = "ERROR"
        meta.error = f"{type(last_err).__name__}: {last_err}" if last_err else "Unknown error"
        meta.latency_ms = round((time.time() - start) * 1000, 1)
        meta.end_time = datetime.utcnow().isoformat() + "Z"
        return {
            "answer": "Agent error. See metadata.",
            "metadata": asdict(meta),
            "error": meta.error,
        }

    def get_state(self, thread_id: str):
        return self.agent.get_state({"configurable": {"thread_id": thread_id}})
