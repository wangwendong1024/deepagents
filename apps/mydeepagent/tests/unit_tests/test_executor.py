from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import Column, MetaData, String, Table, create_engine, insert, select
from sqlalchemy.exc import OperationalError

from mydeepagent.catalog import Catalog, ColumnSpec, TableSpec
from mydeepagent.contracts import ActionPlan, HookDecision, RunContext, SQLStatement
from mydeepagent.executor import Executor
from mydeepagent.hooks import HookRegistry


@pytest.fixture
def setup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'business.sqlite'}")
    metadata = MetaData()
    orders = Table(
        "orders",
        metadata,
        Column("id", String, primary_key=True),
        Column("tenant_id", String, nullable=False),
        Column("state", String),
    )
    metadata.create_all(engine)
    catalog = Catalog(
        version="v1",
        tables={
            "orders": TableSpec(
                columns={name: ColumnSpec(writable=True) for name in ("id", "tenant_id", "state")},
                operations=("select", "insert", "update"),
                tenant_column="tenant_id",
            )
        },
    )
    hooks = HookRegistry()
    executor = Executor(engine, catalog, hooks, max_rows=2)
    executor.initialize_storage()
    context = RunContext(tenant_id="t1", actor_id="worker", event_id="e1", permissions=("query", "compose", "apply"))
    yield executor, context, orders, hooks
    executor.close()


def create_plan(*, state="open", action_id="create-order", identity="1"):
    return ActionPlan(
        kind="apply",
        action_id=action_id,
        catalog_version="v1",
        statements=(
            SQLStatement(
                sql="INSERT INTO orders (id,tenant_id,state) VALUES (:id,:tenant,:state)",
                params={"id": identity, "tenant": "t1", "state": state},
                expected_rows=1,
            ),
        ),
    )


def rows(executor, orders):
    with executor.engine.connect() as connection:
        return connection.execute(select(orders)).mappings().all()


def test_write_and_same_action_replay_are_atomic(setup):
    executor, context, orders, _ = setup
    first = executor.execute(context, create_plan())
    second = executor.execute(context, create_plan())
    assert first.status == "succeeded"
    assert first.affected_rows == (1,)
    assert second.status == "replayed"
    assert len(rows(executor, orders)) == 1


def test_changed_plan_cannot_reuse_committed_slot(setup):
    executor, context, orders, _ = setup
    executor.execute(context, create_plan())
    result = executor.execute(context, create_plan(state="closed"))
    assert result.code == "idempotency_conflict"
    assert rows(executor, orders)[0]["state"] == "open"


@pytest.mark.parametrize("stage", ["before_synthesize", "before_sql", "before_commit"])
def test_hook_denial_never_persists_business_write_or_reserves_slot(setup, stage):
    executor, context, orders, hooks = setup
    hooks.register(stage, lambda payload: HookDecision(outcome="deny", code="business_rule", message="Blocked."))
    assert executor.execute(context, create_plan()).status == "denied"
    assert rows(executor, orders) == []
    executor.hooks = HookRegistry()
    assert executor.execute(context, create_plan()).status == "succeeded"


def test_precommit_sees_own_write_through_restricted_read_view(setup):
    executor, context, orders, hooks = setup
    seen = []

    def check(payload):
        seen.extend(payload.transaction.query("SELECT id, state FROM orders WHERE tenant_id=:tenant", {"tenant": "t1"}))
        return HookDecision(outcome="deny", code="rollback", message="Test rollback.")

    hooks.register("before_commit", check)
    assert executor.execute(context, create_plan()).status == "denied"
    assert seen == [{"id": "1", "state": "open"}]
    assert rows(executor, orders) == []


def test_precommit_cannot_run_sql_write_through_read_view(setup):
    executor, context, orders, hooks = setup

    def forbidden(payload):
        payload.transaction.query("DELETE FROM orders", {})
        return HookDecision()

    hooks.register("before_commit", forbidden)
    assert executor.execute(context, create_plan()).code == "hook_error"
    assert rows(executor, orders) == []


def test_observer_failure_does_not_change_commit_or_cause_duplicate(setup):
    executor, context, orders, hooks = setup

    def broken(payload):
        raise RuntimeError("private details must never escape")

    hooks.register("after_execute", broken, name="audit")
    result = executor.execute(context, create_plan())
    assert result.status == "succeeded"
    assert result.observer_errors == ("audit:hook_error",)
    assert "private" not in result.model_dump_json()
    assert executor.execute(context, create_plan()).status == "replayed"
    assert len(rows(executor, orders)) == 1


def test_replay_obeys_current_hook_policy(setup):
    executor, context, orders, hooks = setup
    executor.execute(context, create_plan())
    hooks.register("before_sql", lambda payload: HookDecision(outcome="deny", code="revoked"))
    assert executor.execute(context, create_plan()).status == "denied"
    assert len(rows(executor, orders)) == 1


def test_failed_second_statement_rolls_back_first_and_ledger(setup):
    executor, context, orders, _ = setup
    first = create_plan().statements[0]
    invalid = SQLStatement(
        sql="UPDATE orders SET state=:state WHERE tenant_id=:tenant AND id=:id",
        params={"state": "closed", "tenant": "t1", "id": "absent"},
        expected_rows=1,
    )
    plan = create_plan().model_copy(update={"statements": (first, invalid)})
    assert executor.execute(context, plan).code == "row_expectation_failed"
    assert rows(executor, orders) == []
    assert executor.execute(context, create_plan()).status == "succeeded"


def test_row_limit_rolls_back_entire_update(setup):
    executor, context, orders, _ = setup
    with executor.engine.begin() as connection:
        connection.execute(insert(orders), [{"id": str(i), "tenant_id": "t1", "state": "open"} for i in range(3)])
    plan = ActionPlan(
        kind="apply",
        action_id="close",
        catalog_version="v1",
        statements=(
            SQLStatement(
                sql="UPDATE orders SET state=:state WHERE tenant_id=:tenant",
                params={"state": "closed", "tenant": "t1"},
            ),
        ),
    )
    assert executor.execute(context, plan).code == "row_limit_exceeded"
    assert {row["state"] for row in rows(executor, orders)} == {"open"}


def test_concurrent_same_action_writes_once(setup):
    executor, context, orders, _ = setup
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: executor.execute(context, create_plan()), range(4)))
    assert sorted(item.status for item in results) == ["replayed", "replayed", "replayed", "succeeded"]
    assert len(rows(executor, orders)) == 1


def test_query_and_parameters_preserve_data_and_tenant_isolation(setup):
    executor, context, orders, _ = setup
    value = "'; DROP TABLE orders; --"
    executor.execute(context, create_plan(state=value))
    with executor.engine.begin() as connection:
        connection.execute(insert(orders).values(id="2", tenant_id="t2", state=value))
    query = ActionPlan(
        kind="query",
        action_id="inspect",
        catalog_version="v1",
        statements=(
            SQLStatement(
                sql="SELECT id, state FROM orders WHERE tenant_id=:tenant AND state=:state",
                params={"tenant": "t1", "state": value},
            ),
        ),
    )
    result = executor.execute(context, query)
    assert result.rows == ({"id": "1", "state": value},)
    assert len(rows(executor, orders)) == 2


def test_missing_permission_or_stale_catalog_cannot_write(setup):
    executor, context, orders, _ = setup
    read_only = context.model_copy(update={"permissions": ("query",)})
    assert executor.execute(read_only, create_plan()).code == "capability_denied"
    assert (
        executor.execute(context, create_plan().model_copy(update={"catalog_version": "old"})).code
        == "catalog_version_mismatch"
    )
    assert rows(executor, orders) == []


def test_compose_is_draft_only_and_requires_lineage(setup):
    executor, context, orders, _ = setup
    plan = ActionPlan(kind="compose", action_id="draft", catalog_version="v1", data={"state": "open"})
    assert executor.execute(context, plan).code == "invalid_composition"
    result = executor.execute(context, plan.model_copy(update={"lineage": {"state": "derived:alarm-rule-v1"}}))
    assert result.status == "succeeded"
    assert result.data == {"state": "open"}
    assert rows(executor, orders) == []


def test_lost_commit_ack_returns_unknown_then_reconciles_without_duplicate(setup, monkeypatch):
    executor, context, orders, _ = setup
    original = executor.engine.dialect.do_commit

    def lose_ack(connection):
        original(connection)
        raise OperationalError("secret connection details", {}, RuntimeError("private"))

    monkeypatch.setattr(executor.engine.dialect, "do_commit", lose_ack)
    result = executor.execute(context, create_plan())
    assert result.status == "unknown"
    assert result.code == "commit_unknown"
    assert "secret" not in result.model_dump_json()
    monkeypatch.setattr(executor.engine.dialect, "do_commit", original)
    assert executor.execute(context, create_plan()).status == "replayed"
    assert len(rows(executor, orders)) == 1


def test_lexical_sql_error_is_structured_and_does_not_echo_input(setup):
    executor, context, orders, _ = setup
    plan = ActionPlan(
        kind="query",
        action_id="read",
        catalog_version="v1",
        statements=(SQLStatement(sql="SELECT id FROM orders WHERE tenant_id='secret-unterminated"),),
    )
    result = executor.execute(context, plan)
    assert result.status == "revise"
    assert result.code == "invalid_sql"
    assert "secret" not in result.model_dump_json()
    assert rows(executor, orders) == []


def test_unexpected_database_error_does_not_echo_secret_parameter(setup):
    executor, context, orders, _ = setup
    executor.execute(context, create_plan())
    result = executor.execute(context, create_plan(action_id="different-slot", state="private-secret"))
    assert result.status == "failed"
    assert result.code == "database_error"
    assert "private-secret" not in result.model_dump_json()
    assert len(rows(executor, orders)) == 1
