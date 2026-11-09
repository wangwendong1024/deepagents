"""Local synthetic smoke test; no cloud calls and no external database access."""

from sqlalchemy import Column, MetaData, String, Table, create_engine

from mydeepagent.catalog import Catalog, ColumnSpec, TableSpec
from mydeepagent.contracts import ActionPlan, HookDecision, RunContext, SQLStatement, ToolResult
from mydeepagent.executor import Executor
from mydeepagent.hooks import HookPayload, HookRegistry


def _demo_catalog() -> Catalog:
    return Catalog(
        version="demo-v1",
        tables={
            "demo_items": TableSpec(
                description="Synthetic items used only for local executor checks.",
                columns={name: ColumnSpec(writable=True) for name in ("id", "tenant_id", "status")},
                operations=("select", "insert", "update"),
                tenant_column="tenant_id",
            )
        },
    )


def _check_demo_status(payload: HookPayload) -> HookDecision:
    if payload.transaction is None:
        return HookDecision(outcome="deny", code="missing_transaction")
    rows = payload.transaction.query(
        "SELECT status FROM demo_items WHERE tenant_id = :tenant AND id = :id",
        {"tenant": payload.context.tenant_id, "id": "item-1"},
    )
    if rows and rows[0]["status"] not in {"open", "closed"}:
        return HookDecision(outcome="deny", code="invalid_status", message="Synthetic status rule rejected the change.")
    return HookDecision()


def run_demo() -> dict[str, ToolResult]:
    """Prove commit, duplicate replay, and hook rollback against an in-memory database."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    Table(
        "demo_items",
        metadata,
        Column("id", String(128), primary_key=True),
        Column("tenant_id", String(128), nullable=False),
        Column("status", String(32), nullable=False),
    )
    metadata.create_all(engine)
    hooks = HookRegistry()
    hooks.register("before_commit", _check_demo_status)
    executor = Executor(engine, _demo_catalog(), hooks)
    try:
        executor.initialize_storage()
        return _demo_actions(executor)
    finally:
        executor.close()


def _demo_actions(executor: Executor) -> dict[str, ToolResult]:
    context = RunContext(
        tenant_id="demo", actor_id="demo", event_id="demo-event", permissions=("query", "compose", "apply")
    )
    insert = ActionPlan(
        action_id="create-item",
        kind="apply",
        catalog_version="demo-v1",
        statements=(
            SQLStatement(
                sql="INSERT INTO demo_items (id, tenant_id, status) VALUES (:id, :tenant, :status)",
                params={"id": "item-1", "tenant": "demo", "status": "open"},
                expected_rows=1,
            ),
        ),
    )
    invalid = ActionPlan(
        action_id="invalid-update",
        kind="apply",
        catalog_version="demo-v1",
        statements=(
            SQLStatement(
                sql="UPDATE demo_items SET status = :status WHERE tenant_id = :tenant AND id = :id",
                params={"status": "invalid", "tenant": "demo", "id": "item-1"},
                expected_rows=1,
            ),
        ),
    )
    read = ActionPlan(
        action_id="verify-item",
        kind="query",
        catalog_version="demo-v1",
        statements=(
            SQLStatement(sql="SELECT id, status FROM demo_items WHERE tenant_id = :tenant", params={"tenant": "demo"}),
        ),
    )
    return {
        "insert": executor.execute(context, insert),
        "duplicate": executor.execute(context, insert),
        "rejected_update": executor.execute(context, invalid),
        "verification": executor.execute(context, read),
    }
