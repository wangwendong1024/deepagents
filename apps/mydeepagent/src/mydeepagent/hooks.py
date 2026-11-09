"""Fail-closed synchronous gates with isolated snapshots and best-effort observers."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from mydeepagent.contracts import ActionPlan, HookDecision, RunContext, Scalar, Stage, ToolResult


class ReadView(Protocol):
    """Read-only facade on the executor's current transaction."""

    def query(self, sql: str, params: dict[str, Scalar]) -> tuple[dict[str, JsonValue], ...]:
        """Run a catalog-authorized query inside the active transaction.

        Args:
            sql: Parameterized SELECT accepted by the SQL policy.
            params: Bound values, including the trusted tenant when required.
        """
        ...


@dataclass(frozen=True)
class HookPayload:
    """A detached plan snapshot and, at commit, a transaction read facade."""

    stage: Stage
    context: RunContext
    plan: ActionPlan
    result: ToolResult | None = None
    transaction: ReadView | None = None


Hook = Callable[[HookPayload], HookDecision]


class HookRegistry:
    """Run trusted local plugins in order; exceptions and mutations reject actions."""

    def __init__(self) -> None:
        """Create an empty registry; mandatory executor policy is separate."""
        self._hooks: dict[Stage, list[tuple[str, Hook]]] = {}

    def register(self, stage: Stage, hook: Hook, *, name: str | None = None) -> None:
        """Register a synchronous operator-owned hook before serving events.

        Args:
            stage: Interception point or after_execute observer.
            hook: Local function, never model-loaded code.
            name: Safe identifier included in diagnostics.
        """
        if stage not in {"before_synthesize", "before_sql", "before_commit", "after_execute"}:
            msg = "Unknown hook stage."
            raise ValueError(msg)
        if not callable(hook):
            msg = "Hooks must be callable."
            raise TypeError(msg)
        if inspect.iscoroutinefunction(hook) or inspect.iscoroutinefunction(hook.__call__):
            msg = "Hooks must be synchronous; do not await network/model calls in transactions."
            raise TypeError(msg)
        self._hooks.setdefault(stage, []).append((name or getattr(hook, "__name__", type(hook).__name__), hook))

    def evaluate(
        self,
        stage: Stage,
        context: RunContext,
        plan: ActionPlan,
        *,
        result: ToolResult | None = None,
        transaction: ReadView | None = None,
    ) -> HookDecision:
        """Stop at the first non-allow response or broken hook.

        Args:
            stage: Gate being evaluated.
            context: Trusted event identity.
            plan: Exact executor-owned plan.
            result: Tentative results, available before commit.
            transaction: Restricted reads on the current transaction.
        """
        for name, hook in self._hooks.get(stage, []):
            decision = self._call(name, hook, HookPayload(stage, context, plan, result, transaction))
            if decision.outcome != "allow":
                return decision
        return HookDecision()

    def observe(self, context: RunContext, plan: ActionPlan, result: ToolResult) -> tuple[str, ...]:
        """Report observer failures without changing a committed result.

        Args:
            context: Trusted event identity.
            plan: Attempted plan, including rejected actions.
            result: Actual terminal result of this attempt.
        """
        errors = []
        for name, hook in self._hooks.get("after_execute", []):
            decision = self._call(name, hook, HookPayload("after_execute", context, plan, result))
            if decision.outcome != "allow":
                errors.append(f"{name}:{decision.code}")
        return tuple(errors)

    @staticmethod
    def _call(name: str, hook: Hook, payload: HookPayload) -> HookDecision:
        snapshot = HookPayload(
            payload.stage,
            payload.context.model_copy(deep=True),
            payload.plan.model_copy(deep=True),
            payload.result.model_copy(deep=True) if payload.result else None,
            payload.transaction,
        )
        try:
            decision = hook(snapshot)
            if inspect.isawaitable(decision):
                if inspect.iscoroutine(decision):
                    decision.close()
                return HookDecision(outcome="deny", code="hook_async", message=f"Hook {name} returned async work.")
            if not isinstance(decision, HookDecision):
                return HookDecision(outcome="deny", code="hook_invalid", message=f"Hook {name} returned no decision.")
            if snapshot.plan.digest() != payload.plan.digest():
                return HookDecision(outcome="deny", code="hook_mutation", message=f"Hook {name} mutated its plan.")
            return decision
        except Exception:
            return HookDecision(outcome="deny", code="hook_error", message=f"Hook {name} failed; execution stopped.")
