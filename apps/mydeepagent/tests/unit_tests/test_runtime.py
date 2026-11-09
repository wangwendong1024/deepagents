import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import Column, MetaData, String, Table, create_engine, select

from mydeepagent.catalog import Catalog, ColumnSpec, TableSpec
from mydeepagent.config import ExecutionSettings
from mydeepagent.contracts import ActionPlan, HookDecision, RunContext, SQLStatement
from mydeepagent.demo import run_demo
from mydeepagent.executor import Executor
from mydeepagent.hooks import HookRegistry
from mydeepagent.runtime import Event, HarnessRuntime
from mydeepagent.tools import ToolSession


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def database(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'runtime.db').as_posix()}")
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
    executor.initialize_storage()
    yield executor, catalog, items
    executor.close()


def plan(action_id="insert-item", tenant="host-tenant"):
    return ActionPlan(
        action_id=action_id,
        kind="apply",
        catalog_version="v1",
        statements=(
            SQLStatement(
                sql="INSERT INTO items (id, tenant) VALUES (:id, :tenant)",
                params={"id": action_id, "tenant": tenant},
                expected_rows=1,
            ),
        ),
    )


def model_for(*plans):
    return ScriptedModel(
        responses=[
            *(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sql_apply",
                            "args": {"plan": item.model_dump(mode="json")},
                            "id": f"call-{index}",
                            "type": "tool_call",
                        }
                    ],
                )
                for index, item in enumerate(plans)
            ),
            AIMessage(content="The model claims it finished."),
        ]
    )


def test_real_agent_tool_executor_path_commits_then_replays(database):
    executor, catalog, items = database
    runtime = HarnessRuntime(
        executor, catalog, ExecutionSettings(tenant_id="host-tenant", writes_enabled=True), model_for(plan())
    )
    event = Event(event_id="alarm-001", payload={"tenant_id": "untrusted-tenant", "actor_id": "admin"})
    first = runtime.handle_event(event)
    second = runtime.handle_event(event)
    assert first.state == "model_finished"
    assert first.actions[0].result.status == "succeeded"
    assert second.actions[0].result.status == "replayed"
    assert first.business_verified is False
    with executor.engine.connect() as conn:
        assert conn.execute(select(items)).all() == [("insert-item", "host-tenant")]


def test_transaction_hook_denial_stops_subsequent_model_writes(database):
    executor, catalog, items = database
    hooks = HookRegistry()
    hooks.register("before_commit", lambda payload: HookDecision(outcome="deny", code="business_rule"))
    executor.hooks = hooks
    runtime = HarnessRuntime(
        executor,
        catalog,
        ExecutionSettings(tenant_id="host-tenant", writes_enabled=True),
        model_for(plan(), plan("second-item")),
    )
    report = runtime.handle_event(Event(event_id="alarm-001", payload={}))
    assert [record.result.status for record in report.actions] == ["denied", "needs_input"]
    with executor.engine.connect() as conn:
        assert conn.execute(select(items)).all() == []


def test_default_write_grant_is_disabled(database):
    executor, catalog, items = database
    runtime = HarnessRuntime(executor, catalog, ExecutionSettings(tenant_id="host-tenant"), model_for(plan()))
    report = runtime.handle_event(Event(event_id="alarm-001", payload={}))
    assert report.actions[0].result.code == "capability_denied"
    with executor.engine.connect() as conn:
        assert conn.execute(select(items)).all() == []


def test_tool_budget_and_kind_are_host_enforced(database):
    executor, catalog, items = database
    context = RunContext(tenant_id="host-tenant", actor_id="host", event_id="one", permissions=("query", "apply"))
    session = ToolSession(executor, catalog, context, ExecutionSettings(max_tool_calls=1))
    tools = {tool.name: tool for tool in session.tools}
    assert tools["catalog_search"].invoke({})["version"] == "v1"
    result = tools["sql_apply"].invoke({"plan": plan().model_dump(mode="json")})
    assert result["code"] == "run_stopped"
    another = ToolSession(executor, catalog, context, ExecutionSettings())
    result = next(tool for tool in another.tools if tool.name == "sql_query").invoke(
        {"plan": plan().model_dump(mode="json")}
    )
    assert result["code"] == "tool_kind_mismatch"
    with executor.engine.connect() as conn:
        assert conn.execute(select(items)).all() == []


def test_revision_budget_cannot_be_reset_by_new_action_id(database):
    executor, catalog, _ = database
    context = RunContext(tenant_id="host-tenant", actor_id="host", event_id="one", permissions=("apply",))
    session = ToolSession(executor, catalog, context, ExecutionSettings(max_revisions=0))
    apply = next(tool for tool in session.tools if tool.name == "sql_apply")
    invalid = plan().model_copy(update={"catalog_version": "old"})
    assert apply.invoke({"plan": invalid.model_dump(mode="json")})["status"] == "revise"
    assert apply.invoke({"plan": plan("new-id").model_dump(mode="json")})["status"] == "needs_input"


def test_event_cannot_override_trusted_identity():
    with pytest.raises(ValueError):
        Event.model_validate({"event_id": "one", "payload": {}, "permissions": ["apply"]})


def test_offline_demo_proves_rollback_and_replay():
    results = run_demo()
    assert results["insert"].status == "succeeded"
    assert results["duplicate"].status == "replayed"
    assert results["rejected_update"].status == "denied"
    assert results["verification"].rows == ({"id": "item-1", "status": "open"},)
    assert "invalid" not in json.dumps(results["verification"].model_dump())
