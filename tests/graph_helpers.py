"""Stand-ins shared by the graph tests, so no test needs the database or a real model.

ScriptedLLM replaces LLMClient: each call returns the next scripted reply (or raises it)
and keeps its arguments. text_reply and call_reply build those replies. make_tools gives
fake tools named like the real ones. run_graph runs the whole graph with them.
"""

import json
from types import SimpleNamespace

from support_agent.graph import build_graph
from support_agent.llm import LLMUnavailableError
from support_agent.request_context import bind_request_context, get_request_context
from support_agent.state import new_state
from support_agent.tools import gated_write


class ScriptedLLM:
    """Stands in for LLMClient. Each call returns, or raises, the next scripted reply.
    The arguments of every call are kept.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, model, input, prompt=None, **options):
        self.calls.append({"model": model, "input": input, "prompt": prompt, **options})
        reply = self.replies[len(self.calls) - 1]
        if isinstance(reply, LLMUnavailableError):
            get_request_context().evidence.append(
                {"type": "model_degraded", "outcome": "escalate", "models_tried": [model]}
            )
        if isinstance(reply, Exception):
            raise reply
        return reply


def text_reply(text):
    return SimpleNamespace(output_text=text, output=[SimpleNamespace(type="message")])


def call_reply(call_id, name, arguments):
    """A model reply that asks for one tool call. `arguments` is the JSON text."""
    item = SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)
    return SimpleNamespace(output_text="", output=[item])


def several_calls_reply(*calls):
    """A model reply that asks for several tool calls at once. Each call is a
    (call_id, tool name) pair, and every call has no arguments.
    """
    items = []
    for call_id, name in calls:
        items.append(
            SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments="{}")
        )
    return SimpleNamespace(output_text="", output=items)


def answer(family="hr", risk="read", language="en"):
    return json.dumps({"family": family, "risk": risk, "language": language})


CHUNK = "vacation-policy#entitlement#0"


CLASSIFIED = text_reply(answer("hr", "read", "en"))


def _add_tool_result(tool, summary, ref):
    get_request_context().evidence.append(
        {"type": "tool_result", "tool": tool, "summary": summary, "ref": ref}
    )


def make_tools(tool_calls):
    """Tools named like the real ones, with no database. Each call they receive is added
    to `tool_calls` as (name, arguments). The fake `create_ticket` acts like the real one:
    a title it has seen before gets the same action id, a new title gets the next id.
    """
    action_ids: dict[str, int] = {}

    def search_policies(query: str) -> str:
        """Search the policies."""
        tool_calls.append(("search_policies", {"query": query}))
        _add_tool_result("search_policies", "1 chunks found", [CHUNK])
        return json.dumps({"results": [{"chunk_id": CHUNK}]})

    def get_leave_balance(year: int = 2026) -> str:
        """Get the balance."""
        tool_calls.append(("get_leave_balance", {"year": year}))
        _add_tool_result("get_leave_balance", "18.0 of 30.0 days left", "leave_balances:x")
        return json.dumps({"remaining_days": 18.0})

    @gated_write
    def create_ticket(title: str) -> str:
        """Propose a ticket."""
        tool_calls.append(("create_ticket", {"title": title}))
        action_id = action_ids.setdefault(title, 5 + len(action_ids))
        _add_tool_result("create_ticket", "proposed a ticket", f"pending_actions:{action_id}")
        return json.dumps({"proposed": True, "action_id": action_id})

    return [search_policies, get_leave_balance, create_ticket]


def run_graph(llm, text="hi", saved_records=None, **graph_options):
    """Run the wired graph in a bound request. Returns the final state, the tools' log
    and the request's evidence.

    The Decision Record is not written to the database. It is added to `saved_records`,
    a list you can pass in to read it afterwards. The policy version is None, because
    reading it needs the database. `graph_options` go to build_graph and replace these
    defaults, for example `max_tool_calls=2` or `get_policy_version=lambda: "2026-09.1"`.
    """
    tool_calls = []
    if saved_records is None:
        saved_records = []
    options = {
        "save_record": saved_records.append,
        "get_policy_version": lambda: None,
        **graph_options,
    }
    graph = build_graph(llm, "small", "main", make_tools(tool_calls), **options)
    with bind_request_context("emp_001", "req-1", "rest") as ctx:
        result = graph.invoke(new_state(text))
    return result, tool_calls, ctx.evidence
