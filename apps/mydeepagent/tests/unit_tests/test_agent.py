"""Exercise the real Deep Agents graph with deterministic model tool calls."""

import json
from collections.abc import Sequence

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import PrivateAttr

from mydeepagent.agent import build_agent

SKILL_PATH = "/skills/business/SKILL.md"
SKILL_BODY = "---\nname: business\ndescription: Apply a business event safely.\n---\nUse sql_query before applying."


class ToolModel(FakeMessagesListChatModel):
    _advertised: list[str] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools: Sequence[BaseTool | dict[str, object]], **kwargs: object):
        self._advertised = [item.name if isinstance(item, BaseTool) else str(item.get("name")) for item in tools]
        return self


def model_calls(name, args):
    return ToolModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1", "type": "tool_call"}]),
            AIMessage(content="finished"),
        ]
    )


def test_allowed_business_tool_executes_and_dangerous_tools_are_hidden():
    called = []

    @tool
    def sql_query(value: str) -> str:
        """Read an authorized business fact."""
        called.append(value)
        return "verified fact"

    model = model_calls("sql_query", {"value": "device"})
    graph = build_agent(model, [sql_query])
    result = graph.invoke({"messages": [{"role": "user", "content": "handle event"}]}, {"recursion_limit": 10})
    assert called == ["device"]
    assert set(model._advertised) == {"sql_query", "read_file"}
    assert any(
        isinstance(message, ToolMessage) and message.content == "verified fact" for message in result["messages"]
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("task", {"description": "Run a shell", "subagent_type": "general-purpose"}),
        ("execute", {"command": "echo forbidden"}),
        ("write_file", {"file_path": SKILL_PATH, "content": "replace trusted rules"}),
        ("read_file", {"file_path": "C:/secret.env"}),
        ("read_file", {"file_path": "/skills/../secret.env"}),
        ("read_file", {"file_path": [SKILL_PATH]}),
        ("unregistered_mcp", {}),
    ],
)
def test_model_cannot_call_unregistered_or_builtin_tools(name, args):
    graph = build_agent(model_calls(name, args), [], skills={SKILL_PATH: SKILL_BODY})
    result = graph.invoke({"messages": [{"role": "user", "content": "handle event"}]}, {"recursion_limit": 10})
    messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(messages) == 1
    assert json.loads(messages[0].content)["code"] == "tool_not_authorized"
    assert result["files"][SKILL_PATH]["content"] == SKILL_BODY


def test_skill_can_be_read_from_virtual_state():
    graph = build_agent(model_calls("read_file", {"file_path": SKILL_PATH}), [], skills={SKILL_PATH: SKILL_BODY})
    result = graph.invoke({"messages": [{"role": "user", "content": "read the skill"}]}, {"recursion_limit": 10})
    messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert "Use sql_query before applying" in messages[0].content


async def test_async_graph_denies_unknown_tool():
    graph = build_agent(model_calls("execute", {"command": "echo forbidden"}), [])
    result = await graph.ainvoke({"messages": [{"role": "user", "content": "handle event"}]}, {"recursion_limit": 10})
    messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert json.loads(messages[0].content)["code"] == "tool_not_authorized"


@pytest.mark.parametrize(
    "path", ["C:/private/SKILL.md", "a/skills/x/SKILL.md", "/skills/../SKILL.md", "//skills/x/SKILL.md"]
)
def test_host_paths_cannot_be_mounted_as_skills(path):
    with pytest.raises(ValueError, match="Skill paths"):
        build_agent(model_calls("read_file", {}), [], skills={path: "secret"})


def test_business_tool_cannot_replace_builtin():
    @tool
    def execute(command: str) -> str:
        """An intentionally invalid tool registration."""
        return command

    with pytest.raises(ValueError, match="must not replace"):
        build_agent(model_calls("execute", {}), [execute])
