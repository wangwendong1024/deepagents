"""Operator-authored semantic metadata and explicit database capability rules."""

import re
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from mydeepagent.contracts import Contract


class ColumnSpec(Contract):
    """Column meaning and independent read/write grants."""

    description: str = ""
    readable: bool = True
    writable: bool = False


class TableSpec(Contract):
    """Allowed operations and optional trusted tenant key for one table."""

    description: str = ""
    columns: dict[str, ColumnSpec]
    operations: tuple[Literal["select", "insert", "update"], ...] = ("select",)
    tenant_column: str | None = None

    @model_validator(mode="after")
    def validate_columns(self) -> Self:
        """Require unqualified identifiers and an explicit tenant column."""
        for name in self.columns:
            _validate_identifier(name)
        if self.tenant_column is not None and self.tenant_column not in self.columns:
            msg = "tenant_column must reference a declared column."
            raise ValueError(msg)
        return self


class Catalog(Contract):
    """Versioned grants; empty tables grants no database access."""

    version: str = Field(min_length=1)
    tables: dict[str, TableSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tables(self) -> Self:
        """Reserve internal tables and reject SQL fragments as identifiers."""
        for name in self.tables:
            _validate_identifier(name)
            if name.lower().startswith("mydeepagent_"):
                msg = "Internal mydeepagent_ tables cannot be granted to agents."
                raise ValueError(msg)
        return self


def _validate_identifier(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        msg = "Catalog identifiers must be unqualified ASCII SQL identifiers."
        raise ValueError(msg)


def load_catalog(path: Path) -> Catalog:
    """Load semantic metadata without connecting to the business database.

    Args:
        path: Trusted, version-controlled JSON catalog path.
    """
    return Catalog.model_validate_json(path.read_text(encoding="utf-8"))
