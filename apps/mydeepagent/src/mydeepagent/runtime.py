"""Single-event runtime shared by local replay and a future durable MQTT ingress."""

import json
from collections.abc import Mapping
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import Field, JsonValue

from mydeepagent.agent import build_agent
from mydeepagent.catalog import Catalog
from mydeepagent.config import ExecutionSettings
from mydeepagent.contracts import Contract, RunContext
from mydeepagent.executor import Executor
from mydeepagent.tools import ActionRecord, ToolSession


class Event(Contract):
    """Untrusted event envelope; tenant, actor, and capabilities cannot be supplied here."""

    event_id: str = Field(min_length=1, max_length=128)
    payload: dict[str, JsonValue]


class RunReport(Contract):
    """Separate actual actions from model prose; business acceptance requires real rules."""

    event_id: str
    state: str
    agent_text: str = ""
    actions: tuple[ActionRecord, ...] = ()
    business_verified: bool = False


def load_skills(directory: Path) -> dict[str, str]:
    """Read only operator-selected SKILL.md files into virtual, read-only skills.

    Args:
        directory: Trusted skills directory, never derived from a message.
    """
    return {
        f"/skills/{path.parent.name}/SKILL.md": path.read_text(encoding="utf-8")
        for path in directory.glob("*/SKILL.md")
    }


class HarnessRuntime:
    """Create a fresh graph and tool budget for each event."""

    def __init__(
        self,
        executor: Executor,
        catalog: Catalog,
        settings: ExecutionSettings,
        model: BaseChatModel,
        *,
        skills: Mapping[str, str] | None = None,
    ) -> None:
        """Assemble runtime dependencies without opening a model request.

        Args:
            executor: Controlled database gateway.
            catalog: Operator-owned semantic catalog.
            settings: Trusted authority, goal, and budgets.
            model: Configured provider adapter or network-free test model.
            skills: Trusted skill documents keyed by virtual paths.
        """
        self._executor = executor
        self._catalog = catalog
        self._settings = settings
        self._model = model
        self._skills = skills

    def handle_event(self, event: Event) -> RunReport:
        """Run one event and retain observed actions even if the model later fails.

        Args:
            event: Validated envelope containing untrusted business data.
        """
        permissions = ("query", "compose", "apply") if self._settings.writes_enabled else ("query", "compose")
        context = RunContext(
            tenant_id=self._settings.tenant_id,
            actor_id=self._settings.actor_id,
            event_id=event.event_id,
            permissions=permissions,
        )
        session = ToolSession(self._executor, self._catalog, context, self._settings)
        agent = build_agent(self._model, session.tools, skills=self._skills)
        content = json.dumps(
            {"goal": self._settings.goal, "event": event.model_dump(), "tenant_id": context.tenant_id},
            ensure_ascii=False,
        )
        try:
            output = agent.invoke(
                {"messages": [{"role": "user", "content": content}]},
                config={"recursion_limit": self._settings.recursion_limit},
            )
        except Exception:
            return RunReport(event_id=event.event_id, state="interrupted", actions=tuple(session.records))
        return RunReport(
            event_id=event.event_id,
            state="model_finished",
            agent_text=str(output["messages"][-1].content),
            actions=tuple(session.records),
        )
