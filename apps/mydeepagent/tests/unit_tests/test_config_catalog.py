import json

import pytest
from pydantic import ValidationError
from sqlalchemy import event

from mydeepagent import config
from mydeepagent.catalog import Catalog, ColumnSpec, TableSpec, load_catalog
from mydeepagent.config import DatabaseSettings, Settings, build_engine, load_settings


def test_loading_config_does_not_resolve_secrets_or_connect(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        '[database]\nurl_env = "TEST_DATABASE_URL"\n[model]\nmodel = "configured-model"\n', encoding="utf-8"
    )
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Configuration loading tried to construct a database engine")

    monkeypatch.setattr(config, "create_engine", forbidden)
    settings = load_settings(path)

    assert settings.database.url_env == "TEST_DATABASE_URL"
    assert settings.model.model == "configured-model"
    assert settings.execution.writes_enabled is False


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": True},
        {"database": {"url": "sqlite:///:memory:"}},
        {"model": {"api_key": "do-not-store-key"}},
        {"execution": {"max_tool_calls": 0}},
        {"execution": {"max_rows": 0}},
    ],
)
def test_config_rejects_unknown_or_invalid_fields(payload):
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


def test_missing_database_environment_is_actionable(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="TEST_DATABASE_URL"):
        build_engine(DatabaseSettings(url_env="TEST_DATABASE_URL"))


def test_invalid_database_url_diagnostic_does_not_echo_credentials(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "not-a-database-url:password=hidden")
    with pytest.raises(ValueError) as failure:
        build_engine(DatabaseSettings(url_env="TEST_DATABASE_URL"))
    assert "hidden" not in str(failure.value)
    assert failure.value.__cause__ is None


def test_engine_construction_is_lazy_and_hides_parameters(tmp_path, monkeypatch):
    path = tmp_path / "business.db"
    monkeypatch.setenv("TEST_DATABASE_URL", f"sqlite:///{path.as_posix()}")
    engine = build_engine(DatabaseSettings(url_env="TEST_DATABASE_URL"))
    try:
        assert not path.exists()
        assert engine.hide_parameters is True
        connections = []
        event.listen(engine, "connect", lambda connection, record: connections.append(True))
        assert connections == []
    finally:
        engine.dispose()


def test_catalog_defaults_to_no_grants_and_read_only_columns():
    catalog = Catalog(version="v1")
    column = ColumnSpec(description="Operator-written meaning")
    table = TableSpec(columns={"id": column})

    assert catalog.tables == {}
    assert table.operations == ("select",)
    assert column.readable is True
    assert column.writable is False


@pytest.mark.parametrize(
    "name", ["jobs; DROP TABLE jobs", "public.jobs", "jobs--", "mydeepagent_actions", "MYDEEPAGENT_LEDGER"]
)
def test_catalog_rejects_fragments_qualified_names_and_internal_tables(name):
    with pytest.raises(ValidationError):
        Catalog(version="v1", tables={name: TableSpec(columns={"id": ColumnSpec()})})


def test_catalog_rejects_unknown_column_and_unlisted_operation():
    with pytest.raises(ValidationError, match="tenant_column"):
        TableSpec(columns={"id": ColumnSpec()}, tenant_column="tenant_id")
    with pytest.raises(ValidationError):
        TableSpec.model_validate({"columns": {"id": {}}, "operations": ["delete"]})
    with pytest.raises(ValidationError):
        TableSpec(columns={"id; DROP TABLE jobs": ColumnSpec()})


def test_loading_catalog_preserves_comments_and_version(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "version": "business-2026-10-28",
                "tables": {
                    "work_orders": {
                        "description": "设备维修工单",
                        "columns": {"tenant_id": {"description": "租户标识"}},
                        "tenant_column": "tenant_id",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    catalog = load_catalog(path)

    assert catalog.version == "business-2026-10-28"
    assert catalog.tables["work_orders"].description == "设备维修工单"
    assert catalog.tables["work_orders"].columns["tenant_id"].description == "租户标识"
