"""Bind business tools to trusted identity and a per-event execution budget."""

from threading import Lock
from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import JsonValue

from mydeepagent.catalog import Catalog
from mydeepagent.config import ExecutionSettings
from mydeepagent.contracts import ActionPlan, Contract, RunContext, ToolResult
from mydeepagent.executor import Executor


class ActionRecord(Contract):
    """Actual tool outcome reported separately from model-generated text."""

    action_id: str
    kind: str
    result: ToolResult


class ToolSession:
    """One event's tools; all identity and grants are closed over by the host."""

    def __init__(self, executor: Executor, catalog: Catalog, context: RunContext, limits: ExecutionSettings) -> None:
        """Bind a controlled executor to an isolated, bounded tool session.

        Args:
            executor: The only holder of database execution capability.
            catalog: Trusted metadata and grants.
            context: Host-provided identity, never accepted as tool input.
            limits: Tool and repair limits for this event.
        """
        self._executor = executor
        self._catalog = catalog.model_copy(deep=True)
        self._context = context
        self._limits = limits
        self._lock = Lock()
        self._calls = 0
        self._revisions = 0
        self._halted = False
        self.records: list[ActionRecord] = []
        self.tools = self._build_tools()

    def _reserve(self) -> bool:
        if self._halted or self._calls >= self._limits.max_tool_calls:
            self._halted = True
            return False
        self._calls += 1
        return True

    def _execute(self, kind: Literal["query", "compose", "apply"], plan: ActionPlan) -> dict[str, JsonValue]:
        with self._lock:
            result = self._dispatch(kind, plan)
            self.records.append(ActionRecord(action_id=plan.action_id, kind=kind, result=result))
            if result.status == "revise":
                self._revisions += 1
                self._halted = self._revisions > self._limits.max_revisions
            if result.status in {"unknown", "needs_input", "denied"}:
                self._halted = True
            return result.model_dump(mode="json")

    def _dispatch(self, kind: Literal["query", "compose", "apply"], plan: ActionPlan) -> ToolResult:
        if not self._reserve():
            return ToolResult(
                status="needs_input", code="run_stopped", message="Execution stopped or budget exhausted."
            )
        if plan.kind != kind:
            return ToolResult(status="denied", code="tool_kind_mismatch", message="Plan kind does not match this tool.")
        return self._executor.execute(self._context, plan)

    def _metadata(self, query: str, *, exact: bool) -> dict[str, JsonValue]:
        with self._lock:
            if not self._reserve():
                return {"status": "needs_input", "code": "run_stopped"}
            tables: dict[str, JsonValue] = {
                name: spec.model_dump(mode="json")
                for name, spec in self._catalog.tables.items()
                if (name == query if exact else query.lower() in f"{name} {spec.description}".lower())
            }
            return {"version": self._catalog.version, "tables": tables}

    def _build_tools(self) -> tuple[BaseTool, ...]:
        @tool
        def catalog_search(query: str = "") -> dict[str, JsonValue]:
            """Search authorized tables and meanings; never discovers ungranted database objects."""
            return self._metadata(query, exact=False)

        @tool
        def catalog_describe(table: str) -> dict[str, JsonValue]:
            """Read the exact authorized table schema, column meanings, and operations."""
            return self._metadata(table, exact=True)

        @tool
        def sql_query(plan: ActionPlan) -> dict[str, JsonValue]:
            """Execute one guarded SELECT with kind=query and named bound parameters."""
            return self._execute("query", plan)

        @tool
        def data_compose(plan: ActionPlan) -> dict[str, JsonValue]:
            """Validate kind=compose data and lineage with before_synthesize; does not write business data."""
            return self._execute("compose", plan)

        @tool
        def sql_apply(plan: ActionPlan) -> dict[str, JsonValue]:
            """Apply kind=apply INSERT/UPDATE atomically through all gates; keep action_id stable on retries."""
            return self._execute("apply", plan)

        @tool
        def business_verify(plan: ActionPlan) -> dict[str, JsonValue]:
            """Observe business state with a guarded kind=query plan; rows are evidence, not automatic acceptance."""
            return self._execute("query", plan)

        tools = (catalog_search, catalog_describe, sql_query, data_compose, sql_apply, business_verify)
        for business_tool in tools:
            business_tool.handle_validation_error = "Invalid tool arguments; use the declared schema."
        return tools
