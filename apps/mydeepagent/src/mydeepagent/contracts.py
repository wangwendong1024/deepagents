"""Versioned contracts shared by tools, hooks, and the executor."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Scalar = str | int | float | bool | None
Stage = Literal["before_synthesize", "before_sql", "before_commit", "after_execute"]


class Contract(BaseModel):
    """Reject unknown fields and attribute replacement at the trust boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RunContext(Contract):
    """Identity supplied by trusted ingress, never by model tool arguments."""

    tenant_id: str = Field(min_length=1, max_length=128)
    actor_id: str = Field(min_length=1, max_length=128)
    event_id: str = Field(min_length=1, max_length=128)
    permissions: tuple[Literal["query", "compose", "apply"], ...] = ("query", "compose")


class SQLStatement(Contract):
    """A single parameterized statement with optional affected-row expectation."""

    sql: str = Field(min_length=1, max_length=16000)
    params: dict[str, Scalar] = Field(default_factory=dict)
    expected_rows: int | None = Field(default=None, ge=0)


class ActionPlan(Contract):
    """One stable business action; action_id survives retries of an event."""

    action_id: str = Field(min_length=1, max_length=128)
    kind: Literal["query", "compose", "apply"]
    catalog_version: str = Field(min_length=1)
    statements: tuple[SQLStatement, ...] = ()
    data: dict[str, JsonValue] = Field(default_factory=dict)
    lineage: dict[str, str] = Field(default_factory=dict)

    def digest(self) -> str:
        """Hash all plan fields to bind the idempotency slot to exact content."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HookDecision(Contract):
    """Structured gate response; only allow may continue execution."""

    outcome: Literal["allow", "revise", "deny", "needs_input"] = "allow"
    code: str = "ok"
    message: str = "Allowed."


class ToolResult(Contract):
    """Observed execution status, independent of the model's completion claim."""

    status: Literal["succeeded", "replayed", "revise", "denied", "needs_input", "failed", "unknown"]
    code: str = "ok"
    message: str = ""
    rows: tuple[dict[str, JsonValue], ...] = ()
    affected_rows: tuple[int, ...] = ()
    data: dict[str, JsonValue] = Field(default_factory=dict)
    observer_errors: tuple[str, ...] = ()


def blocked_result(decision: HookDecision) -> ToolResult:
    """Convert a hook rejection to a tool-visible repair or stop result.

    Args:
        decision: A non-allow decision from the hook chain.
    """
    status = "denied" if decision.outcome == "deny" else decision.outcome
    if status == "allow":
        msg = "Cannot turn an allow decision into a rejection."
        raise ValueError(msg)
    return ToolResult(status=status, code=decision.code, message=decision.message)
