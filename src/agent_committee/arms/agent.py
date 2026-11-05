"""Arm 3: hand-built LangGraph reference agent. The model drives its own tools.

    think (model, may call tools) -> tools -> think ... -> decide (model, no tools)

The linear layer's shortlist is the starting point, but the model writes its own searches and may
pick a label the shortlist missed. Rules:
- An answer before any tool call is sent back once with SEND_BACK, then taken as given.
- A case costs at most AGENT_MAX_TOOL_TURNS + 1 model calls; `decide` must answer, with no tools.
- A malformed tool call ends the case with pred=None, which scores as wrong.
Tool-call ids from the model are random, so replayed messages use `call_{i}` to keep cache keys
stable between runs.
"""

from collections import Counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_committee.arms.agent_tools import TOOLS, ToolBox
from agent_committee.arms.config import AGENT_MAX_TOOL_TURNS, MODEL_DIGEST
from agent_committee.arms.labels import LabelBook
from agent_committee.arms.linear import LinearLayer
from agent_committee.arms.parsing import PromptTooLong, guarded_chat, parse_answer
from agent_committee.arms.retrieval import Retriever
from agent_committee.llm import LLMClient, LLMResult
from agent_committee.schemas import ArmName, ArmPrediction, Case

THINK_MAX_TOKENS = 128
ANSWER_MAX_TOKENS = 32
MAX_CALLS_PER_TURN = 3
SEND_BACK = "Do not answer yet. Use your tools first: call search_examples with your own words."
DECIDE_INSTRUCTION = "Reply now with the single best label only, spelled exactly as shown."


class MalformedToolCall(ValueError):
    pass


def build_system_prompt(shortlist: tuple[str, ...]) -> str:
    return (
        "You classify a banking customer query into exactly one intent label. "
        "Likely candidate labels, most likely first:\n" + "\n".join(shortlist) + "\n"
        "Check before you answer: call search_examples with your own search words, and call "
        "read_label_definition to compare labels. The right label may be missing from the "
        "candidates, so search again in other words if the results look wrong. Then reply with "
        "one label only, spelled exactly as shown, and nothing else."
    )


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    pending: list[dict[str, Any]]
    n_calls: int  # tool calls so far, numbers the normalised ids
    model_calls: int
    used_tool: bool
    own_first_tool: bool  # the model called a tool without being sent back
    sent_back: bool
    retry: bool
    malformed: bool
    answer: str | None
    done: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def _normalize_calls(raw: list[dict[str, Any]], start: int) -> list[dict[str, Any]]:
    calls = []
    for i, call in enumerate(raw[:MAX_CALLS_PER_TURN]):
        function = call.get("function") if isinstance(call, dict) else None
        if (
            not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not isinstance(function.get("arguments"), str)
        ):
            raise MalformedToolCall(repr(call))
        calls.append(
            {
                "id": f"call_{start + i}",
                "type": "function",
                "function": {"name": function["name"], "arguments": function["arguments"]},
            }
        )
    return calls


class AgentArm:
    name = ArmName.AGENT

    def __init__(self, client: LLMClient, book: LabelBook, layer: LinearLayer) -> None:
        self._client = client
        self._book = book
        self._layer = layer
        self._tools: ToolBox | None = None
        self._graph = self._build_graph()
        self.stats: Counter[str] = Counter()

    def fit(self, pool: list[Case]) -> None:
        self._tools = ToolBox(Retriever(pool), self._book, self._layer)

    @staticmethod
    def _usage(state: AgentState, result: LLMResult) -> dict[str, Any]:
        return {
            "model_calls": state["model_calls"] + 1,
            "prompt_tokens": state["prompt_tokens"] + result.prompt_tokens,
            "completion_tokens": state["completion_tokens"] + result.completion_tokens,
            "latency_s": state["latency_s"] + result.latency_s,
        }

    def _think(self, state: AgentState) -> dict[str, Any]:
        result = guarded_chat(
            self._client,
            state["messages"],
            arm=self.name.value,
            max_tokens=THINK_MAX_TOKENS,
            tools=TOOLS,
        )
        update = {**self._usage(state, result), "retry": False}
        raw_calls = result.message.get("tool_calls") or []
        if raw_calls:
            try:
                calls = _normalize_calls(raw_calls, state["n_calls"])
            except MalformedToolCall:
                return {**update, "malformed": True, "answer": None, "done": True}
            assistant = {"role": "assistant", "content": result.content, "tool_calls": calls}
            return {
                **update,
                "messages": [*state["messages"], assistant],
                "pending": calls,
                "n_calls": state["n_calls"] + len(calls),
                "used_tool": True,
                "own_first_tool": state["own_first_tool"] or not state["sent_back"],
            }
        assistant = {"role": "assistant", "content": result.content}
        if not state["used_tool"] and not state["sent_back"]:
            send_back = {"role": "user", "content": SEND_BACK}
            return {
                **update,
                "messages": [*state["messages"], assistant, send_back],
                "sent_back": True,
                "retry": True,
            }
        answer = parse_answer(result, self._book)
        if answer is not None:
            return {**update, "answer": answer, "done": True}
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
        return {"messages": [*state["messages"], *outputs], "pending": []}

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
    def _more_turns(state: AgentState) -> bool:
        return state["model_calls"] < AGENT_MAX_TOOL_TURNS

    def _after_think(self, state: AgentState) -> str:
        if state["done"]:
            return END
        if state["pending"]:
            return "tools"
        return "think" if state["retry"] and self._more_turns(state) else "decide"

    def _after_tools(self, state: AgentState) -> str:
        return "think" if self._more_turns(state) else "decide"

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("think", self._think)
        graph.add_node("tools", self._run_tools)
        graph.add_node("decide", self._decide)
        graph.add_edge(START, "think")
        graph.add_conditional_edges("think", self._after_think, ["think", "tools", "decide", END])
        graph.add_conditional_edges("tools", self._after_tools, ["think", "decide"])
        graph.add_edge("decide", END)
        return graph.compile()

    def predict(self, case_id: str, text: str) -> ArmPrediction:
        if self._tools is None:
            raise RuntimeError("fit() first")
        shortlist = self._layer.decide(text).shortlist
        try:
            final = self._invoke(shortlist, text)
        except PromptTooLong:
            # the conversation outgrew the model's context: scored wrong and counted, not a crash
            self.stats.update(cases=1, too_long_cases=1)
            return ArmPrediction(case_id=case_id, arm=self.name, pred=None, too_long=True)
        self.stats.update(
            cases=1,
            model_calls=final["model_calls"],
            own_tool_cases=int(final["own_first_tool"]),
            sent_back_cases=int(final["sent_back"]),
            malformed_cases=int(final["malformed"]),
        )
        return ArmPrediction(
            case_id=case_id,
            arm=self.name,
            pred=final["answer"],
            prompt_tokens=final["prompt_tokens"],
            completion_tokens=final["completion_tokens"],
            latency_s=final["latency_s"],
            own_tool_calls=final["n_calls"],
            sent_back=final["sent_back"],
            malformed=final["malformed"],
        )

    def _invoke(self, shortlist: tuple[str, ...], text: str) -> dict[str, Any]:
        return self._graph.invoke(
            {
                "messages": [
                    {"role": "system", "content": build_system_prompt(shortlist)},
                    {"role": "user", "content": text},
                ],
                "pending": [],
                "n_calls": 0,
                "model_calls": 0,
                "used_tool": False,
                "own_first_tool": False,
                "sent_back": False,
                "retry": False,
                "malformed": False,
                "answer": None,
                "done": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_s": 0.0,
            }
        )


def make_arm(
    pool: list[Case], book: LabelBook, layer: LinearLayer, client: LLMClient | None = None
) -> AgentArm:
    client = client or LLMClient(phase="build", model_digest=MODEL_DIGEST)
    arm = AgentArm(client, book, layer)
    arm.fit(pool)
    return arm
