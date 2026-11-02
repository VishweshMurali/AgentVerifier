"""Arm 3: hand-built LangGraph reference agent.

    retrieve -> think (may call tools, at most AGENT_MAX_TOOL_TURNS times) -> tools -> ... -> decide

`retrieve` is deterministic and costs no model call: left to choose, the model skipped its tools
on every one of 100 tune cases, so the first search always runs and arrives as a tool result.
`decide` has no tools, so a case costs at most AGENT_MAX_TOOL_TURNS + 1 model calls. If `think`
answers directly with a valid label the case ends there, so the agent pays for no extra call.
Tool-call ids from the model are random, so replayed messages use `call_{i}` to keep cache keys
stable between runs.
"""

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_committee.arms.agent_tools import SEARCH, TOOLS, ToolBox
from agent_committee.arms.config import AGENT_MAX_TOOL_TURNS, MODEL_DIGEST
from agent_committee.arms.labels import LabelBook
from agent_committee.arms.parsing import guarded_chat, parse_answer
from agent_committee.arms.retrieval import Retriever
from agent_committee.llm import LLMClient
from agent_committee.schemas import ArmName, ArmPrediction, Case

THINK_MAX_TOKENS = 128
ANSWER_MAX_TOKENS = 32
MAX_CALLS_PER_TURN = 3
DECIDE_INSTRUCTION = "Reply now with the single best label only, spelled exactly as listed."


def build_system_prompt(book: LabelBook) -> str:
    return (
        "You classify a banking customer query into exactly one intent label. "
        "Similar labelled queries are retrieved for you first; weigh them. Call "
        "read_label_definition to compare labels you are unsure about, or search_examples "
        "again with other words. Then reply with one label from the list below, spelled "
        "exactly as shown, and nothing else. Never invent a new label.\nLabels:\n"
        + "\n".join(book.labels)
    )


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    pending: list[dict[str, Any]]
    n_calls: int  # tool calls so far, numbers the normalised ids
    turns: int
    answer: str | None
    done: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def _normalize_calls(raw: list[dict[str, Any]], start: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"call_{start + i}",
            "type": "function",
            "function": {
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            },
        }
        for i, call in enumerate(raw[:MAX_CALLS_PER_TURN])
    ]


class AgentArm:
    name = ArmName.AGENT

    def __init__(self, client: LLMClient, book: LabelBook) -> None:
        self._client = client
        self._book = book
        self._system = build_system_prompt(book)
        self._tools: ToolBox | None = None
        self._graph = self._build_graph()

    def fit(self, pool: list[Case]) -> None:
        self._tools = ToolBox(Retriever(pool), self._book)

    def _usage(self, state: AgentState, result: Any) -> dict[str, Any]:
        return {
            "prompt_tokens": state["prompt_tokens"] + result.prompt_tokens,
            "completion_tokens": state["completion_tokens"] + result.completion_tokens,
            "latency_s": state["latency_s"] + result.latency_s,
        }

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        assert self._tools is not None
        query = state["messages"][-1]["content"]
        call = {
            "id": "call_0",
            "type": "function",
            "function": {"name": SEARCH, "arguments": json.dumps({"query": query})},
        }
        output = self._tools.search_examples(query)
        return {
            "messages": [
                *state["messages"],
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_0", "content": output},
            ],
            "n_calls": 1,
        }

    def _think(self, state: AgentState) -> dict[str, Any]:
        result = guarded_chat(
            self._client,
            state["messages"],
            arm=self.name.value,
            max_tokens=THINK_MAX_TOKENS,
            tools=TOOLS,
        )
        update = self._usage(state, result)
        raw_calls = result.message.get("tool_calls") or []
        if raw_calls:
            calls = _normalize_calls(raw_calls, state["n_calls"])
            assistant = {"role": "assistant", "content": result.content, "tool_calls": calls}
            return {
                **update,
                "messages": [*state["messages"], assistant],
                "pending": calls,
                "n_calls": state["n_calls"] + len(calls),
            }
        answer = parse_answer(result, self._book)
        if answer is not None:
            return {**update, "answer": answer, "done": True}
        assistant = {"role": "assistant", "content": result.content}
        return {**update, "messages": [*state["messages"], assistant]}

    def _run_tools(self, state: AgentState) -> dict[str, Any]:
        assert self._tools is not None
        outputs = [
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": self._tools.run(call["function"]["name"], call["function"]["arguments"]),
            }
            for call in state["pending"]
        ]
        return {
            "messages": [*state["messages"], *outputs],
            "pending": [],
            "turns": state["turns"] + 1,
        }

    def _decide(self, state: AgentState) -> dict[str, Any]:
        messages = [*state["messages"], {"role": "user", "content": DECIDE_INSTRUCTION}]
        result = guarded_chat(
            self._client, messages, arm=self.name.value, max_tokens=ANSWER_MAX_TOKENS
        )
        return {
            **self._usage(state, result),
            "answer": parse_answer(result, self._book),
            "done": True,
        }

    @staticmethod
    def _after_think(state: AgentState) -> str:
        if state["done"]:
            return END
        return "tools" if state["pending"] else "decide"

    @staticmethod
    def _after_tools(state: AgentState) -> str:
        return "decide" if state["turns"] >= AGENT_MAX_TOOL_TURNS else "think"

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("think", self._think)
        graph.add_node("tools", self._run_tools)
        graph.add_node("decide", self._decide)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "think")
        graph.add_conditional_edges("think", self._after_think, ["tools", "decide", END])
        graph.add_conditional_edges("tools", self._after_tools, ["think", "decide"])
        graph.add_edge("decide", END)
        return graph.compile()

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        if self._tools is None:
            raise RuntimeError("fit() first")
        final = self._graph.invoke(
            {
                "messages": [
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": text},
                ],
                "pending": [],
                "n_calls": 0,
                "turns": 0,
                "answer": None,
                "done": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_s": 0.0,
            }
        )
        return ArmPrediction(
            case_id=case_id,
            arm=self.name,
            pred=final["answer"],
            prompt_tokens=final["prompt_tokens"],
            completion_tokens=final["completion_tokens"],
            latency_s=final["latency_s"],
        )


def make_arm(pool: list[Case], book: LabelBook, client: LLMClient | None = None) -> AgentArm:
    client = client or LLMClient(phase="build", model_digest=MODEL_DIGEST)
    arm = AgentArm(client, book)
    arm.fit(pool)
    return arm
