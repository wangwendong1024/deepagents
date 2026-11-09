"""A later model failure cannot erase a committed action or trigger blind replay."""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import Column, MetaData, String, Table, create_engine, select

from mydeepagent.catalog import Catalog, ColumnSpec, TableSpec
from mydeepagent.config import ExecutionSettings
from mydeepagent.contracts import ActionPlan, SQLStatement
from mydeepagent.executor import Executor
from mydeepagent.runtime import Event, HarnessRuntime


class FailsAfterToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        if self.i == 1:
            raise RuntimeError("sensitive-provider-error-body")
        return super()._generate(*args, **kwargs)


class ReplayModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_model_failure_after_commit_preserves_evidence_and_safe_replay(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'failure.db').as_posix()}")
    metadata = MetaData()
    items = Table("items", metadata, Column("id", String, primary_key=True), Column("tenant", String, nullable=False))
    metadata.create_all(engine)
    catalog = Catalog(
        version="v1",
        tables={
            "items": TableSpec(
                columns={"id": ColumnSpec(writable=True), "tenant": ColumnSpec(writable=True)},
                operations=("select", "insert"),
                tenant_column="tenant",
            )
        },
    )
    executor = Executor(engine, catalog)
    try:
        executor.initialize_storage()
        _assert_failed_run_and_replay(executor, catalog, items)
    finally:
        executor.close()


def _assert_failed_run_and_replay(executor, catalog, items):
    plan = ActionPlan(
        action_id="create-item",
        kind="apply",
        catalog_version="v1",
        statements=(
            SQLStatement(
                sql="INSERT INTO items (id, tenant) VALUES (:id, :tenant)",
                params={"id": "item-1", "tenant": "trusted"},
                expected_rows=1,
            ),
        ),
    )
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "sql_apply", "args": {"plan": plan.model_dump(mode="json")}, "id": "action", "type": "tool_call"}
        ],
    )
    settings = ExecutionSettings(tenant_id="trusted", writes_enabled=True)
    event = Event(event_id="alarm-001", payload={})
    failed = HarnessRuntime(
        executor, catalog, settings, FailsAfterToolModel(responses=[call, AIMessage(content="unreachable")])
    ).handle_event(event)
    assert failed.state == "interrupted"
    assert failed.business_verified is False
    assert len(failed.actions) == 1
    assert failed.actions[0].result.status == "succeeded"
    assert "sensitive-provider-error-body" not in failed.model_dump_json()
    replay = HarnessRuntime(
        executor, catalog, settings, ReplayModel(responses=[call, AIMessage(content="finished")])
    ).handle_event(event)
    assert replay.state == "model_finished"
    assert replay.actions[0].result.status == "replayed"
    with executor.engine.connect() as connection:
        assert connection.execute(select(items)).all() == [("item-1", "trusted")]
