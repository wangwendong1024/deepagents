"""Load trusted TOML configuration; secrets remain in environment variables."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field
from sqlalchemy import Engine, create_engine

from mydeepagent.contracts import Contract


class ModelSettings(Contract):
    """Connection settings for a tool-calling cloud model."""

    provider: Literal["openai", "anthropic"] = "openai"
    model: str = ""
    base_url: str | None = None
    api_key_env: str = "MYDEEPAGENT_MODEL_API_KEY"
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_retries: int = Field(default=1, ge=0, le=3)
    temperature: float | None = None
    use_responses_api: bool = False


class DatabaseSettings(Contract):
    """Database URLs are resolved at execution time and never sent to a model."""

    url_env: str = "MYDEEPAGENT_DATABASE_URL"


class ExecutionSettings(Contract):
    """Trusted identity and finite resource limits for this service instance."""

    tenant_id: str = Field(default="local-demo", min_length=1, max_length=128)
    actor_id: str = Field(default="mydeepagent", min_length=1, max_length=128)
    writes_enabled: bool = False
    max_rows: int = Field(default=100, ge=1, le=1000)
    max_statements: int = Field(default=10, ge=1, le=50)
    max_tool_calls: int = Field(default=24, ge=1, le=100)
    max_revisions: int = Field(default=3, ge=0, le=10)
    recursion_limit: int = Field(default=40, ge=1, le=200)
    goal: str = "Process the event using the catalog and business skill; do not invent missing rules."


class Settings(Contract):
    """Application configuration independent of a particular business schema."""

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    catalog_path: str = "catalog.example.json"
    skills_directory: str = "skills"


def load_settings(path: Path) -> Settings:
    """Read a trusted TOML file and reject unsupported keys.

    Args:
        path: Local configuration selected by the service operator.
    """
    with path.open("rb") as stream:
        return Settings.model_validate(tomllib.load(stream))


def build_engine(settings: DatabaseSettings) -> Engine:
    """Construct a lazy SQLAlchemy engine without revealing connection secrets.

    Args:
        settings: Environment-variable reference containing the database URL.
    """
    url = os.environ.get(settings.url_env)
    if not url:
        msg = f"Set the database URL environment variable {settings.url_env}."
        raise ValueError(msg)
    try:
        engine = create_engine(url, pool_pre_ping=True, hide_parameters=True)
    except Exception:
        msg = "Database configuration or driver is invalid; check the selected URL and installed extra."
        raise ValueError(msg) from None
    if engine.dialect.name not in {"sqlite", "postgresql", "mysql"}:
        engine.dispose()
        msg = "Supported database dialects are sqlite, postgresql, and mysql."
        raise ValueError(msg)
    return engine
