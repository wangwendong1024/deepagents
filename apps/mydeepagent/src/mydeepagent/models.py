"""Explicit cloud model adapters; secrets never enter the Agent state."""

from __future__ import annotations

import os
from typing import NotRequired, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from mydeepagent.config import ModelSettings


class _ModelOptions(TypedDict):
    api_key: SecretStr
    temperature: NotRequired[float]
    timeout: float
    max_retries: int
    base_url: NotRequired[str]
    use_responses_api: NotRequired[bool]
    store: NotRequired[bool]


def build_model(config: ModelSettings) -> BaseChatModel:
    """Construct a configured model without making an inference request.

    Args:
        config: Provider settings and the name of a secret environment variable.

    Returns:
        A tool-capable LangChain model for the business harness.

    Raises:
        ValueError: If configuration or the provider adapter is unavailable.
    """
    if not config.model.strip():
        msg = "Set model.model to your provider's available model identifier."
        raise ValueError(msg)
    if config.provider not in {"openai", "anthropic"}:
        msg = "Supported adapters: openai (including compatible APIs), anthropic."
        raise ValueError(msg)
    if config.provider == "anthropic" and config.use_responses_api:
        msg = "use_responses_api is only valid with the openai adapter."
        raise ValueError(msg)
    secret = os.environ.get(config.api_key_env, "")
    if not secret.strip():
        msg = "The configured model API key environment variable is empty."
        raise ValueError(msg)
    return _initialize(config, SecretStr(secret))


def _initialize(config: ModelSettings, secret: SecretStr) -> BaseChatModel:
    options: _ModelOptions = {
        "api_key": secret,
        "timeout": config.timeout_seconds,
        "max_retries": config.max_retries,
    }
    if config.temperature is not None:
        options["temperature"] = config.temperature
    if config.base_url:
        options["base_url"] = config.base_url
    if config.provider == "openai":
        options["use_responses_api"] = config.use_responses_api
        if config.use_responses_api:
            options["store"] = False
    try:
        return init_chat_model(config.model, model_provider=config.provider, **options)
    except (ImportError, ValueError, TypeError):
        # SDK validation may echo credentials; only expose a fixed diagnostic.
        msg = "Could not initialize the model adapter; check configuration and installed provider dependencies."
        raise ValueError(msg) from None
