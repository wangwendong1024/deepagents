import pytest
from sqlalchemy.dialects import mysql, postgresql, sqlite

from mydeepagent.catalog import Catalog, ColumnSpec, TableSpec
from mydeepagent.contracts import RunContext, SQLStatement
from mydeepagent.sql_policy import SQLPolicy, SQLPolicyError


def catalog():
    return Catalog(
        version="v1",
        tables={
            "orders": TableSpec(
                operations=("select", "insert", "update"),
                tenant_column="tenant_id",
                columns={
                    "id": ColumnSpec(writable=True),
                    "tenant_id": ColumnSpec(writable=True),
                    "state": ColumnSpec(writable=True),
                    "secret": ColumnSpec(readable=False),
                },
            )
        },
    )


@pytest.fixture
def policy():
    return SQLPolicy(catalog(), max_rows=5)


@pytest.fixture
def context():
    return RunContext(tenant_id="t1", actor_id="service", event_id="e1", permissions=("query", "compose", "apply"))


@pytest.mark.parametrize(
    ("sql", "params", "write"),
    [
        ("SELECT id FROM orders WHERE tenant_id=:tenant OR id=:id", {"tenant": "t1", "id": "1"}, False),
        ("SELECT * FROM orders WHERE tenant_id=:tenant", {"tenant": "t1"}, False),
        ("SELECT secret FROM orders WHERE tenant_id=:tenant", {"tenant": "t1"}, False),
        ("SELECT id FROM orders WHERE tenant_id=:tenant", {"tenant": "t2"}, False),
        ("SELECT id FROM orders", {}, False),
        ("SELECT load_extension(:path) FROM orders WHERE tenant_id=:tenant", {"path": "bad", "tenant": "t1"}, False),
        ("SELECT id FROM orders WHERE tenant_id='t1'", {}, False),
        ("SELECT id FROM orders WHERE tenant_id=:tenant; DROP TABLE orders", {"tenant": "t1"}, False),
        ("SELECT id FROM orders WHERE tenant_id=:tenant LIMIT 6", {"tenant": "t1"}, False),
        ("SELECT id FROM orders WHERE tenant_id=:tenant LIMIT -1", {"tenant": "t1"}, False),
        ("SELECT id FROM orders WHERE tenant_id=:tenant LIMIT :limit", {"tenant": "t1", "limit": 2}, False),
        ("SELECT id FROM orders WHERE tenant_id=:tenant", {"tenant": "t1", "extra": "unused"}, False),
        ("SELECT id FROM orders WHERE tenant_id=:tenant UNION SELECT secret FROM orders", {"tenant": "t1"}, False),
        ("SELECT orders.id FROM orders JOIN orders other ON other.id=orders.id", {}, False),
        ("SELECT id INTO outfile FROM orders WHERE tenant_id=:tenant", {"tenant": "t1"}, False),
        ("SELECT id FROM other WHERE tenant_id=:tenant", {"tenant": "t1"}, False),
        ("SELECT id FROM main.orders WHERE tenant_id=:tenant", {"tenant": "t1"}, False),
        ("UPDATE orders SET state=:state", {"state": "open"}, True),
        ("UPDATE orders SET tenant_id=:new WHERE tenant_id=:tenant", {"new": "t2", "tenant": "t1"}, True),
        (
            "UPDATE orders SET state=:state WHERE tenant_id=:tenant RETURNING secret",
            {"state": "x", "tenant": "t1"},
            True,
        ),
        ("INSERT INTO orders (id) VALUES (:id)", {"id": "1"}, True),
        ("INSERT INTO orders (tenant_id,id) VALUES (:tenant,:id),(:tenant,:id)", {"tenant": "t1", "id": "1"}, True),
        ("INSERT OR IGNORE INTO orders (tenant_id,id) VALUES (:tenant,:id)", {"tenant": "t1", "id": "1"}, True),
        ("DELETE FROM orders WHERE tenant_id=:tenant", {"tenant": "t1"}, True),
        ("DROP TABLE orders", {}, True),
        ("PRAGMA writable_schema=ON", {}, False),
        ("SELECT id FROM orders WHERE tenant_id='unterminated", {}, False),
        ("SELECT id FROM orders WHERE tenant_id=:tenant /* unfinished", {"tenant": "t1"}, False),
    ],
)
def test_rejects_unsupported_or_unscoped_sql(policy, context, sql, params, write):
    with pytest.raises(SQLPolicyError):
        policy.validate(context, SQLStatement(sql=sql, params=params), write=write)


@pytest.mark.parametrize(
    ("name", "dialect"),
    [("sqlite", sqlite.dialect()), ("postgresql", postgresql.dialect()), ("mysql", mysql.dialect())],
)
def test_rebuilds_parameterized_sql_for_configured_dialect(context, name, dialect):
    policy = SQLPolicy(catalog(), dialect=name, max_rows=5)
    value = "'; DROP TABLE orders; --"
    statement = policy.validate(
        context,
        SQLStatement(
            sql="SELECT id, state FROM orders WHERE tenant_id=:tenant AND state=:state ORDER BY id DESC LIMIT 3",
            params={"tenant": "t1", "state": value},
        ),
    )
    compiled = str(statement.statement.compile(dialect=dialect))
    assert "DROP TABLE" not in compiled
    assert statement.params["bound_state"] == value


def test_read_only_catalog_cannot_write(context):
    policy = SQLPolicy(Catalog(version="v1", tables={"orders": TableSpec(columns={"id": ColumnSpec()})}))
    with pytest.raises(SQLPolicyError, match="table or operation"):
        policy.validate(
            context, SQLStatement(sql="INSERT INTO orders (id) VALUES (:id)", params={"id": "1"}), write=True
        )


def test_null_predicate_and_parentheses_are_supported(policy, context):
    statement = policy.validate(
        context,
        SQLStatement(sql="SELECT id FROM orders WHERE (tenant_id=:tenant AND state IS NULL)", params={"tenant": "t1"}),
    )
    assert statement.operation == "select"
    assert "IS NULL" in str(statement.statement)


def test_complexity_is_bounded_before_recursive_rebuild(policy, context):
    sql = "SELECT id FROM orders WHERE tenant_id=:tenant" + " AND id=:id" * 100
    with pytest.raises(SQLPolicyError) as caught:
        policy.validate(context, SQLStatement(sql=sql, params={"tenant": "t1", "id": "1"}))
    assert caught.value.code == "sql_too_complex"
