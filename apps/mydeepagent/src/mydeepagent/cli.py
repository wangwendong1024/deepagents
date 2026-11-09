"""Explicit local commands; startup never implicitly changes database schemas."""

import argparse
import json
from pathlib import Path

from mydeepagent.catalog import load_catalog
from mydeepagent.config import build_engine, load_settings
from mydeepagent.demo import run_demo
from mydeepagent.executor import Executor
from mydeepagent.models import build_model
from mydeepagent.runtime import Event, HarnessRuntime, load_skills


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metadata-driven business agent foundation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run offline synthetic commit/replay/rollback checks")
    for command in ("check-config", "init-storage", "replay"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        if command == "replay":
            subparser.add_argument("--event", type=Path, required=True)
    return parser


def _configured(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    base = args.config.resolve().parent
    catalog = load_catalog(base / settings.catalog_path)
    if args.command == "check-config":
        print(json.dumps({"valid": True, "catalog_version": catalog.version, "tables": len(catalog.tables)}))
        return 0
    executor = Executor(
        build_engine(settings.database),
        catalog,
        max_rows=settings.execution.max_rows,
        max_statements=settings.execution.max_statements,
    )
    try:
        if args.command == "init-storage":
            executor.initialize_storage()
            print(json.dumps({"initialized": True, "business_tables_created": False}))
            return 0
        model = build_model(settings.model)
        runtime = HarnessRuntime(
            executor, catalog, settings.execution, model, skills=load_skills(base / settings.skills_directory)
        )
        report = runtime.handle_event(Event.model_validate_json(args.event.read_text(encoding="utf-8")))
        print(report.model_dump_json(indent=2))
        return 0 if report.state == "model_finished" else 1
    finally:
        executor.close()


def main() -> int:
    """Run a local command and return an exit status without logging credentials."""
    args = _parser().parse_args()
    try:
        if args.command == "demo":
            result = run_demo()
            print(json.dumps({name: item.model_dump(mode="json") for name, item in result.items()}, indent=2))
            expected = ("succeeded", "replayed", "denied", "succeeded")
            return 0 if tuple(item.status for item in result.values()) == expected else 1
        return _configured(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": "command_failed",
                    "type": type(error).__name__,
                    "message": "Check local configuration and prerequisites.",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
