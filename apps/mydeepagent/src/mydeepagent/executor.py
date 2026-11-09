"""Enforced tool boundary with short transactions and an atomic action ledger."""

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import date, datetime, time
from decimal import Decimal
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import Column, MetaData, String, Table, Text, insert, select, update
from sqlalchemy.engine import Connection, Engine, RowMapping
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from mydeepagent.catalog import Catalog
from mydeepagent.contracts import ActionPlan, RunContext, Scalar, SQLStatement, ToolResult, blocked_result
from mydeepagent.hooks import HookRegistry
from mydeepagent.sql_policy import SQLPolicy, SQLPolicyError, ValidatedSQL

_METADATA = MetaData()
_ACTIONS = Table(
    "mydeepagent_actions",
    _METADATA,
    Column("action_key", String(64), primary_key=True, comment="SHA-256 of tenant, event, and stable action ID."),
    Column("digest", String(64), nullable=False, comment="Exact plan and trusted authority fingerprint."),
    Column("result_json", Text, nullable=False, comment="Committed result; persisted atomically with business writes."),
    comment="MyDeepAgent-owned idempotency ledger. No foreign keys. Retain throughout event replay horizon.",
)


def _json_scalar(value: object) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        msg = "Query returned a non-finite number that is not valid JSON data."
        raise SQLPolicyError("unsupported_result_type", msg)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (Decimal, UUID)):
        return str(value)
    msg = "Query returned a value that needs an explicit JSON conversion adapter."
    raise SQLPolicyError("unsupported_result_type", msg)


def _rows(rows: Sequence[RowMapping]) -> tuple[dict[str, JsonValue], ...]:
    return tuple({str(key): _json_scalar(value) for key, value in row.items()} for row in rows)


class _TransactionView:
    def __init__(self, connection: Connection, policy: SQLPolicy, context: RunContext) -> None:
        self.__connection = connection
        self.__policy = policy
        self.__context = context

    def query(self, sql: str, params: dict[str, Scalar]) -> tuple[dict[str, JsonValue], ...]:
        """Read through the same policy and transaction without exposing commit methods.

        Args:
            sql: Parameterized catalog SELECT.
            params: Scalar values bound by the SQLAlchemy driver.
        """
        statement = self.__policy.validate(self.__context, SQLStatement(sql=sql, params=params))
        return _rows(self.__connection.execute(statement.statement, statement.params).mappings().all())


class Executor:
    """Execute only validated plans; direct tool calls cannot skip the hooks."""

    def __init__(
        self,
        engine: Engine,
        catalog: Catalog,
        hooks: HookRegistry | None = None,
        *,
        max_rows: int = 100,
        max_statements: int = 10,
    ) -> None:
        """Bind one database, one catalog revision, and trusted local hooks.

        Args:
            engine: Operator-created SQLAlchemy engine; credentials stay here.
            catalog: Versioned table and column grants.
            hooks: Synchronous policy plugins and completion observers.
            max_rows: Query limit and maximum total affected rows in one apply.
            max_statements: Maximum atomic write statements in one action.
        """
        if max_statements < 1:
            msg = "max_statements must be positive."
            raise ValueError(msg)
        self.engine = engine
        self.catalog = catalog.model_copy(deep=True)
        self.hooks = hooks or HookRegistry()
        self.policy = SQLPolicy(self.catalog, dialect=engine.dialect.name, max_rows=max_rows)
        self.max_statements = max_statements

    def initialize_storage(self) -> None:
        """Explicitly create only the internal action ledger, never business tables."""
        _METADATA.create_all(self.engine)

    def close(self) -> None:
        """Dispose this executor's database connections."""
        self.engine.dispose()

    def execute(self, context: RunContext, plan: ActionPlan) -> ToolResult:
        """Validate snapshots, enforce gates, and report observed execution status.

        Args:
            context: Trusted ingress identity; never sourced from model arguments.
            plan: Proposed action whose stable ID survives retries.
        """
        context = RunContext.model_validate(context.model_dump(mode="json"))
        plan = ActionPlan.model_validate(plan.model_dump(mode="json"))
        try:
            result = self._dispatch(context, plan)
        except SQLPolicyError as exc:
            result = ToolResult(status="denied" if exc.denied else "revise", code=exc.code, message=str(exc))
        except SQLAlchemyError:
            result = ToolResult(
                status="failed", code="database_error", message="Database operation failed; inspect server diagnostics."
            )
        errors = self.hooks.observe(context, plan, result)
        return result.model_copy(update={"observer_errors": errors})

    def _dispatch(self, context: RunContext, plan: ActionPlan) -> ToolResult:
        self._validate_plan(context, plan)
        if plan.kind in {"compose", "apply"}:
            decision = self.hooks.evaluate("before_synthesize", context, plan)
            if decision.outcome != "allow":
                return blocked_result(decision)
        if plan.kind == "compose":
            return ToolResult(status="succeeded", code="draft_composed", data=plan.model_copy(deep=True).data)
        statements = tuple(self.policy.validate(context, item, write=plan.kind == "apply") for item in plan.statements)
        decision = self.hooks.evaluate("before_sql", context, plan)
        if decision.outcome != "allow":
            return blocked_result(decision)
        if plan.kind == "query":
            with self.engine.connect() as connection:
                statement = statements[0]
                rows = _rows(connection.execute(statement.statement, statement.params).mappings().all())
                return ToolResult(status="succeeded", code="query_result", rows=rows)
        return self._apply(context, plan, statements)

    def _validate_plan(self, context: RunContext, plan: ActionPlan) -> None:
        if plan.kind not in context.permissions:
            msg = "Trusted ingress has not granted this operation."
            raise SQLPolicyError("capability_denied", msg, denied=True)
        if plan.catalog_version != self.catalog.version:
            msg = "Refresh catalog metadata and rebuild the action with its current version."
            raise SQLPolicyError("catalog_version_mismatch", msg)
        if plan.kind == "compose":
            if (
                plan.statements
                or set(plan.lineage) != set(plan.data)
                or any(not value for value in plan.lineage.values())
            ):
                msg = "Compose accepts data with one nonempty lineage description per field and no SQL."
                raise SQLPolicyError("invalid_composition", msg)
        elif not 1 <= len(plan.statements) <= (1 if plan.kind == "query" else self.max_statements):
            msg = "Query requires one statement; apply requires a nonempty bounded statement list."
            raise SQLPolicyError("invalid_statement_count", msg)

    @staticmethod
    def _identity(context: RunContext, plan: ActionPlan) -> tuple[str, str]:
        key = json.dumps([context.tenant_id, context.event_id, plan.action_id], separators=(",", ":"))
        authority = json.dumps([context.actor_id, sorted(context.permissions), plan.digest()], separators=(",", ":"))
        return hashlib.sha256(key.encode()).hexdigest(), hashlib.sha256(authority.encode()).hexdigest()

    def _apply(self, context: RunContext, plan: ActionPlan, statements: tuple[ValidatedSQL, ...]) -> ToolResult:
        key, digest = self._identity(context, plan)
        with self.engine.connect() as connection:
            transaction = connection.begin()
            try:
                connection.execute(insert(_ACTIONS).values(action_key=key, digest=digest, result_json=""))
            except IntegrityError:
                transaction.rollback()
                return self._replay(connection, key, digest)
            try:
                result = self._write(connection, context, plan, statements)
                if result.status != "succeeded":
                    transaction.rollback()
                    return result
                connection.execute(
                    update(_ACTIONS).where(_ACTIONS.c.action_key == key).values(result_json=result.model_dump_json())
                )
            except Exception:
                transaction.rollback()
                raise
            try:
                transaction.commit()
            except SQLAlchemyError:
                return ToolResult(
                    status="unknown",
                    code="commit_unknown",
                    message=(
                        "Commit acknowledgement failed. Reconcile this same action ID and plan "
                        "before changing or retrying the business action."
                    ),
                )
            return result

    def _write(
        self, connection: Connection, context: RunContext, plan: ActionPlan, statements: tuple[ValidatedSQL, ...]
    ) -> ToolResult:
        affected = []
        for source, statement in zip(plan.statements, statements, strict=True):
            result = connection.execute(statement.statement, statement.params)
            count = result.rowcount
            if count < 0 or sum(affected) + count > self.policy.max_rows:
                msg = "Affected row count is unknown or exceeds the configured transaction limit."
                raise SQLPolicyError("row_limit_exceeded", msg, denied=True)
            if source.expected_rows is not None and count != source.expected_rows:
                msg = "Affected rows did not match the plan; the entire transaction was rolled back."
                raise SQLPolicyError("row_expectation_failed", msg)
            affected.append(count)
        tentative = ToolResult(status="succeeded", code="committed", affected_rows=tuple(affected))
        decision = self.hooks.evaluate(
            "before_commit",
            context,
            plan,
            result=tentative,
            transaction=_TransactionView(connection, self.policy, context),
        )
        return tentative if decision.outcome == "allow" else blocked_result(decision)

    @staticmethod
    def _replay(connection: Connection, key: str, digest: str) -> ToolResult:
        stored = connection.execute(select(_ACTIONS).where(_ACTIONS.c.action_key == key)).mappings().first()
        if stored is None or not stored["result_json"]:
            return ToolResult(
                status="unknown",
                code="reconciliation_required",
                message="No committed ledger result is visible; do not invent a new action ID.",
            )
        if stored["digest"] != digest:
            return ToolResult(
                status="denied",
                code="idempotency_conflict",
                message="This action ID is already committed with different content or authority.",
            )
        result = ToolResult.model_validate_json(stored["result_json"])
        return result.model_copy(update={"status": "replayed", "code": "already_committed"})
