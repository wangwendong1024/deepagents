"""Validate model SQL and rebuild a deliberately small grammar as SQLAlchemy Core."""

import operator
import re
from dataclasses import dataclass
from typing import Literal

import sqlglot
from sqlalchemy import and_, bindparam, column, insert, select, table, update
from sqlalchemy.sql import ColumnElement, Executable, Select, TableClause
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mydeepagent.catalog import Catalog, TableSpec
from mydeepagent.contracts import RunContext, Scalar, SQLStatement

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_COMPARISONS = {
    exp.EQ: operator.eq,
    exp.NEQ: operator.ne,
    exp.GT: operator.gt,
    exp.GTE: operator.ge,
    exp.LT: operator.lt,
    exp.LTE: operator.le,
}


class SQLPolicyError(ValueError):
    """A stable, non-secret rejection suitable for model repair."""

    def __init__(self, code: str, message: str, *, denied: bool = False) -> None:
        """Keep database exceptions and supplied values out of feedback.

        Args:
            code: Machine-readable reason.
            message: Trusted explanation of the supported operation.
            denied: Whether policy forbids the operation outright.
        """
        super().__init__(message)
        self.code = code
        self.denied = denied


@dataclass(frozen=True)
class ValidatedSQL:
    """Executable built from validated nodes, never the original SQL string."""

    statement: Executable
    params: dict[str, Scalar]
    operation: Literal["select", "insert", "update"]


def _only_args(node: exp.Expr, allowed: set[str]) -> None:
    if any(value and key not in allowed for key, value in node.args.items()):
        msg = "This SQL feature is unsupported; use one table and the documented parameterized grammar."
        raise SQLPolicyError("unsupported_sql", msg)


def _name(node: exp.Expr) -> str:
    if not isinstance(node, exp.Identifier) or not _IDENTIFIER.fullmatch(node.name):
        msg = "Identifiers must be simple catalog names."
        raise SQLPolicyError("invalid_identifier", msg)
    _only_args(node, {"this", "quoted"})
    return node.name


class SQLPolicy:
    """Enforce catalog access, tenant predicates, named bindings, and row limits."""

    def __init__(self, catalog: Catalog, *, dialect: str = "sqlite", max_rows: int = 100) -> None:
        """Configure a policy without opening a database connection.

        Args:
            catalog: Trusted versioned schema and capability configuration.
            dialect: SQLAlchemy database dialect name.
            max_rows: Upper bound for one query and a write transaction.
        """
        if dialect not in {"sqlite", "postgresql", "mysql"}:
            msg = "Only SQLite, PostgreSQL, and MySQL dialects are supported."
            raise ValueError(msg)
        if max_rows < 1:
            msg = "max_rows must be positive."
            raise ValueError(msg)
        self.catalog = catalog.model_copy(deep=True)
        self.dialect = "postgres" if dialect == "postgresql" else dialect
        self.max_rows = max_rows

    def validate(self, context: RunContext, statement: SQLStatement, *, write: bool = False) -> ValidatedSQL:
        """Parse and rebuild a statement before it reaches a database driver.

        Args:
            context: Identity supplied by trusted ingress.
            statement: Model SQL with separately supplied scalar parameters.
            write: Whether this is an apply operation rather than a query.

        Raises:
            SQLPolicyError: SQL is unsupported or outside the catalog policy.
        """
        node = self._parse(statement.sql)
        builder = _Builder(self, context, statement.params)
        if isinstance(node, exp.Select) and not write:
            compiled = builder.select(node)
            operation = "select"
        elif isinstance(node, exp.Insert) and write:
            compiled = builder.insert(node)
            operation = "insert"
        elif isinstance(node, exp.Update) and write:
            compiled = builder.update(node)
            operation = "update"
        else:
            msg = "Query accepts SELECT; apply accepts INSERT or UPDATE only."
            raise SQLPolicyError("operation_not_allowed", msg, denied=True)
        if builder.bindings != set(statement.params):
            msg = "Supply exactly the named :parameters used in this statement."
            raise SQLPolicyError("parameters_mismatch", msg)
        bindings = {f"bound_{name}": value for name, value in statement.params.items()}
        return ValidatedSQL(compiled, bindings, operation)

    def _parse(self, sql: str) -> exp.Expr:
        try:
            nodes = sqlglot.parse(sql, read=self.dialect)
        except (SqlglotError, RecursionError) as exc:
            msg = "SQL could not be parsed using the configured database dialect."
            raise SQLPolicyError("invalid_sql", msg) from exc
        if len(nodes) != 1 or nodes[0] is None:
            msg = "Each SQLStatement must contain exactly one statement."
            raise SQLPolicyError("multiple_statements", msg)
        self._check_complexity(nodes[0])
        return nodes[0]

    @staticmethod
    def _check_complexity(root: exp.Expr) -> None:
        pending = [(root, 0)]
        count = 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > 256 or depth > 32:
                msg = "SQL exceeds the foundation grammar complexity limit; split the action into smaller statements."
                raise SQLPolicyError("sql_too_complex", msg)
            pending.extend((child, depth + 1) for child in node.iter_expressions())


class _Builder:
    def __init__(self, policy: SQLPolicy, context: RunContext, params: dict[str, Scalar]) -> None:
        self.policy = policy
        self.context = context
        self.params = params
        self.bindings: set[str] = set()
        self.target: TableClause
        self.spec: TableSpec
        self.tenant_bound = False

    def _table(self, node: exp.Expr, operation: str) -> None:
        if not isinstance(node, exp.Table):
            msg = "A single catalog table is required; subqueries and joins are unsupported."
            raise SQLPolicyError("unsupported_table", msg)
        _only_args(node, {"this"})
        name = _name(node.this)
        spec = self.policy.catalog.tables.get(name)
        if name.lower().startswith("mydeepagent_") or spec is None or operation not in spec.operations:
            msg = "The table or operation is outside the configured catalog policy."
            raise SQLPolicyError("table_not_allowed", msg, denied=True)
        self.spec = spec
        self.target = table(name, *(column(name) for name in spec.columns))

    def _column(self, node: exp.Expr, *, writable: bool = False) -> ColumnElement[object]:
        if not isinstance(node, exp.Column):
            msg = "Use explicit catalog columns; expressions, functions, and wildcards are unsupported."
            raise SQLPolicyError("unsupported_expression", msg)
        _only_args(node, {"this", "table"})
        if node.table and node.table != self.target.name:
            msg = "Column qualification must name the single target table."
            raise SQLPolicyError("invalid_column_scope", msg)
        name = _name(node.this)
        self._column_access(name, writable=writable)
        return self.target.c[name]

    def _column_access(self, name: str, *, writable: bool) -> None:
        spec = self.spec.columns.get(name)
        if spec is None or not (spec.writable if writable else spec.readable):
            msg = "A referenced column is outside the configured read/write policy."
            raise SQLPolicyError("column_not_allowed", msg, denied=True)

    def _parameter(self, node: exp.Expr) -> ColumnElement[object]:
        if not isinstance(node, exp.Placeholder) or not _IDENTIFIER.fullmatch(node.name):
            msg = "Data values must be named :parameters, never literals or SQL expressions."
            raise SQLPolicyError("parameter_required", msg)
        _only_args(node, {"this"})
        if node.name not in self.params:
            msg = "A named SQL parameter has no supplied value."
            raise SQLPolicyError("missing_parameter", msg)
        self.bindings.add(node.name)
        # Prefix internal bind names to avoid SQLAlchemy UPDATE reserved column names.
        return bindparam(f"bound_{node.name}", value=self.params[node.name])

    def _predicate(self, node: exp.Expr) -> ColumnElement[bool]:
        _only_args(node, {"this", "expression"})
        if isinstance(node, exp.Paren):
            return self._predicate(node.this)
        if isinstance(node, exp.And):
            return and_(self._predicate(node.this), self._predicate(node.expression))
        left = self._column(node.this)
        if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
            return left.is_(None)
        comparator = _COMPARISONS.get(type(node))
        if comparator is None:
            msg = "WHERE supports AND and column-to-parameter comparisons or IS NULL."
            raise SQLPolicyError("unsupported_predicate", msg)
        right = self._parameter(node.expression)
        if isinstance(node, exp.EQ) and node.this.name == self.spec.tenant_column:
            self.tenant_bound |= self.params[node.expression.name] == self.context.tenant_id
        return comparator(left, right)

    def _where(self, node: exp.Expr | None, *, required: bool = False) -> ColumnElement[bool] | None:
        if node is None:
            if required or self.spec.tenant_column:
                msg = "A scoped WHERE clause is required for this operation."
                raise SQLPolicyError("where_required", msg)
            return None
        _only_args(node, {"this"})
        predicate = self._predicate(node.this)
        if self.spec.tenant_column and not self.tenant_bound:
            msg = "Bind the tenant column with equality to the trusted run tenant."
            raise SQLPolicyError("tenant_scope_required", msg, denied=True)
        return predicate

    def select(self, node: exp.Select) -> Select:
        _only_args(node, {"expressions", "from_", "where", "order", "limit"})
        source = node.args.get("from_")
        if source is None:
            msg = "SELECT requires a catalog table."
            raise SQLPolicyError("table_required", msg)
        _only_args(source, {"this"})
        self._table(source.this, "select")
        selected = [self._column(item) for item in node.expressions]
        if not selected or len({str(item) for item in selected}) != len(selected):
            msg = "SELECT requires unique, explicit columns."
            raise SQLPolicyError("invalid_projection", msg)
        result = select(*selected).select_from(self.target)
        predicate = self._where(node.args.get("where"))
        if predicate is not None:
            result = result.where(predicate)
        return self._order_limit(result, node)

    def _order_limit(self, result: Select, node: exp.Select) -> Select:
        order = node.args.get("order")
        if order is not None:
            _only_args(order, {"expressions"})
            for item in order.expressions:
                _only_args(item, {"this", "desc", "nulls_first"})
                value = self._column(item.this)
                null_order = value.is_(None)
                result = result.order_by(null_order.desc() if item.args.get("nulls_first") else null_order.asc())
                result = result.order_by(value.desc() if item.args.get("desc") else value.asc())
        limit = self.policy.max_rows
        requested = node.args.get("limit")
        if requested is not None:
            _only_args(requested, {"expression"})
            value = requested.expression
            if not isinstance(value, exp.Literal) or value.is_string or not value.this.isdigit():
                msg = "LIMIT must be a positive integer within max_rows."
                raise SQLPolicyError("invalid_limit", msg)
            limit = int(value.this)
            if not 1 <= limit <= self.policy.max_rows:
                msg = "LIMIT must be a positive integer within max_rows."
                raise SQLPolicyError("invalid_limit", msg)
        return result.limit(limit)

    def insert(self, node: exp.Insert) -> Executable:
        _only_args(node, {"this", "expression"})
        schema = node.this
        values = node.expression
        if not isinstance(schema, exp.Schema) or not isinstance(values, exp.Values) or len(values.expressions) != 1:
            msg = "INSERT requires explicit columns and exactly one VALUES row."
            raise SQLPolicyError("unsupported_insert", msg)
        _only_args(schema, {"this", "expressions"})
        _only_args(values, {"expressions"})
        self._table(schema.this, "insert")
        row = values.expressions[0]
        if not isinstance(row, exp.Tuple):
            msg = "INSERT requires one VALUES tuple."
            raise SQLPolicyError("unsupported_insert", msg)
        _only_args(row, {"expressions"})
        names = [_name(item) for item in schema.expressions]
        if len(names) != len(row.expressions) or len(set(names)) != len(names) or not names:
            msg = "INSERT columns and values must match without duplicate columns."
            raise SQLPolicyError("invalid_insert_values", msg)
        assignments = {}
        for name, value in zip(names, row.expressions, strict=True):
            self._column_access(name, writable=True)
            assignments[name] = self._parameter(value)
            if name == self.spec.tenant_column:
                self.tenant_bound = self.params[value.name] == self.context.tenant_id
        if self.spec.tenant_column and not self.tenant_bound:
            msg = "INSERT must bind its tenant column to the trusted run tenant."
            raise SQLPolicyError("tenant_scope_required", msg, denied=True)
        return insert(self.target).values(assignments)

    def update(self, node: exp.Update) -> Executable:
        _only_args(node, {"this", "expressions", "where"})
        self._table(node.this, "update")
        assignments = {}
        for item in node.expressions:
            if not isinstance(item, exp.EQ):
                msg = "UPDATE SET requires column-to-parameter assignments."
                raise SQLPolicyError("unsupported_update", msg)
            _only_args(item, {"this", "expression"})
            self._column(item.this, writable=True)
            name = item.this.name
            if name == self.spec.tenant_column or name in assignments:
                msg = "Tenant reassignment and duplicate SET columns are forbidden."
                raise SQLPolicyError("invalid_assignment", msg, denied=True)
            assignments[name] = self._parameter(item.expression)
        if not assignments:
            msg = "UPDATE requires at least one assignment."
            raise SQLPolicyError("invalid_assignment", msg)
        predicate = self._where(node.args.get("where"), required=True)
        if predicate is None:
            msg = "UPDATE requires a scoped WHERE clause."
            raise SQLPolicyError("where_required", msg)
        return update(self.target).values(assignments).where(predicate)
