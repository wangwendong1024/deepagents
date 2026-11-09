"""Verify live MyDeepAgent tool execution with an isolated, read-only SQLite fixture.

The two-row fixture exists only for this process and has no logical references or
time fields. Target database foreign keys: none.
Existing database foreign keys found: none.
"""

import argparse
import contextlib
import io
import json
import os
import secrets
import sqlite3
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from unittest.mock import patch

from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.schemas import Run
from pydantic import JsonValue

from mydeepagent.cli import main as application_main
from mydeepagent.config import ModelSettings, load_settings
from mydeepagent.contracts import ActionPlan, SQLStatement
from mydeepagent.runtime import RunReport
from mydeepagent.tools import ActionRecord

TENANT = "inference-smoke"
CATALOG = "inference-smoke-v1"
ACTION = "read-local-smoke-row"


def _plan() -> ActionPlan:
    return ActionPlan(
        action_id=ACTION,
        kind="query",
        catalog_version=CATALOG,
        statements=(
            SQLStatement(
                sql="SELECT id, status FROM demo_items WHERE tenant_id = :tenant AND id = :id",
                params={"tenant": TENANT, "id": "item-1"},
            ),
        ),
    )


def _write_catalog(directory: Path) -> None:
    catalog = {
        "version": CATALOG,
        "tables": {
            "demo_items": {
                "description": "Synthetic integration fixture.",
                "columns": {name: {"readable": True, "writable": False} for name in ("id", "tenant_id", "status")},
                "operations": ["select"],
                "tenant_column": "tenant_id",
            }
        },
    }
    (directory / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")


def _write_database(directory: Path, marker: str) -> None:
    with contextlib.closing(sqlite3.connect(directory / "fixture.db")) as database, database:
        database.execute("CREATE TABLE demo_items (id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, status TEXT NOT NULL)")
        database.execute("INSERT INTO demo_items VALUES (?, ?, ?)", ("item-1", TENANT, marker))
        database.execute("INSERT INTO demo_items VALUES (?, ?, ?)", ("other", "other-tenant", "excluded"))


def _write_config(directory: Path, model: ModelSettings) -> None:
    arguments = json.dumps({"plan": _plan().model_dump(mode="json")})
    goal = (
        "This is a synthetic read-only integration check. Call sql_query exactly once with "
        "the following JSON arguments. Then report the status string from the returned row "
        "exactly as observed. Never guess the status or write anything. Arguments: " + arguments
    )
    model_lines = [f"{key} = {json.dumps(value)}" for key, value in model.model_dump(exclude_none=True).items()]
    lines = [
        'catalog_path = "catalog.json"',
        'skills_directory = "skills"',
        "[model]",
        *model_lines,
        "[execution]",
        f'tenant_id = "{TENANT}"',
        'actor_id = "mydeepagent-integration-smoke"',
        "writes_enabled = false",
        "max_tool_calls = 4",
        "recursion_limit = 16",
        f"goal = {json.dumps(goal)}",
        "",
    ]
    (directory / "config.toml").write_text("\n".join(lines), encoding="utf-8")


def _clear_proxy_and_tracing() -> None:
    for name in tuple(os.environ):
        if name.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            del os.environ[name]
    os.environ.update(NO_PROXY="*", LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false", OTEL_SDK_DISABLED="true")


def _walk_trace(runs: Sequence[Run]) -> Iterator[Run]:
    for run in runs:
        yield run
        yield from _walk_trace(run.child_runs)


def _replay(directory: Path) -> tuple[RunReport, tuple[str, ...]]:
    arguments = [
        "mydeepagent",
        "replay",
        "--config",
        str(directory / "config.toml"),
        "--event",
        str(directory / "event.json"),
    ]
    output = io.StringIO()
    with collect_runs() as trace, contextlib.redirect_stdout(output), patch.object(sys, "argv", arguments):
        exit_code = application_main()
    if exit_code != 0:
        msg = "MyDeepAgent replay failed. Inspect the configured endpoint and server logs."
        raise RuntimeError(msg)
    tools = tuple(run.name for run in _walk_trace(trace.traced_runs) if run.run_type == "tool" and not run.error)
    return RunReport.model_validate_json(output.getvalue()), tools


def _validate_report(report: RunReport, tools: tuple[str, ...], marker: str) -> ActionRecord:
    rows = ({"id": "item-1", "status": marker},)
    matching = [
        action
        for action in report.actions
        if action.action_id == ACTION
        and action.kind == "query"
        and action.result.status == "succeeded"
        and action.result.rows == rows
    ]
    if report.state != "model_finished" or "sql_query" not in tools or len(matching) != 1:
        msg = "The live run did not finish with the expected successful sql_query and fixture row."
        raise RuntimeError(msg)
    if marker not in report.agent_text:
        msg = "The SQL tool succeeded, but the final answer omitted the observed status."
        raise RuntimeError(msg)
    return matching[0]


def _prepare_fixture(directory: Path, model: ModelSettings, marker: str) -> None:
    _write_catalog(directory)
    _write_database(directory, marker)
    _write_config(directory, model)
    event = {"event_id": "local-model-smoke", "payload": {"item_id": "item-1"}}
    (directory / "event.json").write_text(json.dumps(event), encoding="utf-8")


def verify(model_config: Path) -> dict[str, JsonValue]:
    """Run the actual CLI against a live model and confirm its observed SQL result.

    Args:
        model_config: Existing MyDeepAgent configuration selecting the live model.

    Returns:
        Selected evidence from the synthetic run, without credentials or business data.

    Raises:
        RuntimeError: The credential is missing or actual tool execution fails.
    """
    _clear_proxy_and_tracing()
    model = load_settings(model_config).model
    if not os.environ.get(model.api_key_env):
        msg = "Set the model configuration's API key environment variable before running verification."
        raise RuntimeError(msg)
    marker = "fixture-status-" + secrets.token_hex(4)
    with tempfile.TemporaryDirectory(prefix="mydeepagent-replay-") as temporary:
        directory = Path(temporary)
        _prepare_fixture(directory, model, marker)
        database_url = f"sqlite:///{(directory / 'fixture.db').as_posix()}"
        with patch.dict(os.environ, {"MYDEEPAGENT_DATABASE_URL": database_url}):
            report, tools = _replay(directory)
        action = _validate_report(report, tools, marker)
    return {
        "model": model.model,
        "state": report.state,
        "tool": "sql_query",
        "action": action.model_dump(mode="json"),
        "final_answer": report.agent_text,
        "writes_enabled": False,
        "business_verified": report.business_verified,
    }


def main() -> int:
    """Print actual tool evidence selected by an explicit model profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.model_config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
