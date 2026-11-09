import pytest

from mydeepagent.contracts import ActionPlan, HookDecision, RunContext, SQLStatement, ToolResult
from mydeepagent.hooks import HookRegistry


@pytest.fixture
def context():
    return RunContext(tenant_id="tenant-1", actor_id="service", event_id="alarm-1")


@pytest.fixture
def plan():
    return ActionPlan(
        action_id="draft",
        kind="compose",
        catalog_version="v1",
        data={"work_order": {"status": "open"}},
        lineage={"work_order": "event:alarm-1"},
    )


@pytest.mark.parametrize("outcome", ["revise", "deny", "needs_input"])
def test_gate_stops_at_first_non_allow(context, plan, outcome):
    registry = HookRegistry()
    reached = []

    def reject(payload):
        return HookDecision(outcome=outcome, code="business_rule", message="Resolve assignment.")

    def later(payload):
        reached.append(payload)
        return HookDecision()

    registry.register("before_synthesize", reject)
    registry.register("before_synthesize", later)
    decision = registry.evaluate("before_synthesize", context, plan)

    assert decision.outcome == outcome
    assert decision.code == "business_rule"
    assert reached == []


def test_hook_exception_fails_closed_without_exception_text(context, plan):
    registry = HookRegistry()

    def crash(payload):
        raise RuntimeError("password=do-not-return-this")

    registry.register("before_synthesize", crash)
    decision = registry.evaluate("before_synthesize", context, plan)

    assert decision.outcome == "deny"
    assert decision.code == "hook_error"
    assert "do-not-return-this" not in decision.model_dump_json()


@pytest.mark.parametrize("returned", [None, True, {"outcome": "allow"}])
def test_hook_requires_typed_decision(context, plan, returned):
    registry = HookRegistry()
    registry.register("before_synthesize", lambda payload: returned, name="invalid")
    decision = registry.evaluate("before_synthesize", context, plan)

    assert decision.outcome == "deny"
    assert decision.code == "hook_invalid"


def test_nested_plan_mutation_is_rejected_and_original_is_unchanged(context, plan):
    registry = HookRegistry()

    def mutate(payload):
        payload.plan.data["work_order"]["status"] = "closed"
        return HookDecision()

    registry.register("before_synthesize", mutate)
    decision = registry.evaluate("before_synthesize", context, plan)

    assert decision.code == "hook_mutation"
    assert decision.outcome == "deny"
    assert plan.data["work_order"]["status"] == "open"


def test_sql_parameter_mutation_does_not_change_executor_plan(context):
    plan = ActionPlan(
        action_id="lookup",
        kind="query",
        catalog_version="v1",
        statements=(SQLStatement(sql="SELECT id FROM jobs WHERE tenant_id = :tenant", params={"tenant": "tenant-1"}),),
    )
    registry = HookRegistry()

    def mutate(payload):
        payload.plan.statements[0].params["tenant"] = "tenant-2"
        return HookDecision()

    registry.register("before_sql", mutate)
    decision = registry.evaluate("before_sql", context, plan)

    assert decision.code == "hook_mutation"
    assert plan.statements[0].params["tenant"] == "tenant-1"


def test_after_execute_observers_continue_and_preserve_committed_result(context, plan):
    registry = HookRegistry()
    observations = []
    result = ToolResult(status="succeeded", data={"work_order": {"status": "open"}})

    def crash(payload):
        raise ValueError("observer transport unavailable")

    def mutate_result(payload):
        payload.result.data["work_order"]["status"] = "closed"
        return HookDecision(outcome="deny", code="observer_rejected")

    def record(payload):
        observations.append(payload.result)
        return HookDecision()

    registry.register("after_execute", crash)
    registry.register("after_execute", mutate_result)
    registry.register("after_execute", record)
    errors = registry.observe(context, plan, result)

    assert errors == ("crash:hook_error", "mutate_result:observer_rejected")
    assert observations == [result]
    assert result.status == "succeeded"
    assert result.data["work_order"]["status"] == "open"


def test_registration_rejects_unknown_stage_and_async_hook():
    registry = HookRegistry()

    async def asynchronous(payload):
        return HookDecision()

    with pytest.raises(ValueError, match="Unknown hook stage"):
        registry.register("before_execute", lambda payload: HookDecision())
    with pytest.raises(TypeError, match="synchronous"):
        registry.register("before_commit", asynchronous)


def test_registration_rejects_async_callable_object():
    class AsyncHook:
        async def __call__(self, payload):
            return HookDecision()

    with pytest.raises(TypeError, match="synchronous"):
        HookRegistry().register("before_commit", AsyncHook())


def test_unexpected_coroutine_is_closed_without_running(context, plan):
    registry = HookRegistry()
    executed = []

    async def asynchronous():
        executed.append(True)
        return HookDecision()

    registry.register("before_commit", lambda payload: asynchronous(), name="hidden_async")
    decision = registry.evaluate("before_commit", context, plan)

    assert decision.outcome == "deny"
    assert decision.code == "hook_async"
    assert executed == []


def test_synchronous_callable_object_is_a_valid_local_hook(context, plan):
    class BusinessCheck:
        def __call__(self, payload):
            return HookDecision(outcome="needs_input", code="assignment_required")

    registry = HookRegistry()
    registry.register("before_commit", BusinessCheck())
    decision = registry.evaluate("before_commit", context, plan)

    assert decision.outcome == "needs_input"
    assert decision.code == "assignment_required"


def test_commit_hook_receives_tentative_result_and_transaction_view(context, plan):
    registry = HookRegistry()
    tentative = ToolResult(status="succeeded", affected_rows=(1,))

    class TransactionView:
        def query(self, sql, params):
            assert params == {"tenant": "tenant-1"}
            return ({"id": "work-order-1"},)

    def verify(payload):
        assert payload.stage == "before_commit"
        assert payload.result == tentative
        rows = payload.transaction.query("SELECT id FROM jobs WHERE tenant_id = :tenant", {"tenant": "tenant-1"})
        return HookDecision() if rows else HookDecision(outcome="deny", code="missing_order")

    registry.register("before_commit", verify)
    decision = registry.evaluate("before_commit", context, plan, result=tentative, transaction=TransactionView())

    assert decision.outcome == "allow"
