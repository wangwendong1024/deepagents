"""Provider configuration is explicit and secrets stay out of diagnostics."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from mydeepagent.config import ModelSettings
from mydeepagent.models import build_model


def test_missing_model_and_key_fail_before_initialization(monkeypatch):
    monkeypatch.delenv("MYDEEPAGENT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="model.model"):
        build_model(ModelSettings())
    with pytest.raises(ValueError, match="environment variable is empty"):
        build_model(ModelSettings(model="configured-model"))


@pytest.mark.parametrize("responses", [True, False])
def test_openai_compatible_configuration_is_forwarded(monkeypatch, responses):
    monkeypatch.setenv("TEST_MODEL_SECRET", "test-secret-never-log")
    captured = {}
    fake = FakeMessagesListChatModel(responses=[AIMessage(content="done")])

    def initialize(model, **options):
        captured.update({"model": model, **options})
        return fake

    monkeypatch.setattr("mydeepagent.models.init_chat_model", initialize)
    config = ModelSettings(
        provider="openai",
        model="configured-model",
        base_url="https://provider.example/v1",
        api_key_env="TEST_MODEL_SECRET",
        timeout_seconds=12,
        max_retries=2,
        temperature=0,
        use_responses_api=responses,
    )
    assert build_model(config) is fake
    assert captured["model"] == "configured-model"
    assert captured["base_url"] == "https://provider.example/v1"
    assert captured["use_responses_api"] is responses
    if responses:
        assert captured["store"] is False
    else:
        assert "store" not in captured
    assert captured["timeout"] == 12
    assert captured["max_retries"] == 2
    assert captured["temperature"] == 0
    assert isinstance(captured["api_key"], SecretStr)
    assert "test-secret-never-log" not in repr(captured)


def test_anthropic_does_not_receive_openai_options(monkeypatch):
    monkeypatch.setenv("MYDEEPAGENT_MODEL_API_KEY", "dummy")
    captured = {}
    fake = FakeMessagesListChatModel(responses=[AIMessage(content="done")])

    def initialize(model, **options):
        captured.update(options)
        return fake

    monkeypatch.setattr("mydeepagent.models.init_chat_model", initialize)
    assert build_model(ModelSettings(provider="anthropic", model="configured-model")) is fake
    assert captured["model_provider"] == "anthropic"
    assert "use_responses_api" not in captured
    assert "store" not in captured
    assert "temperature" not in captured


def test_adapter_diagnostic_never_echoes_secret(monkeypatch):
    monkeypatch.setenv("MYDEEPAGENT_MODEL_API_KEY", "do-not-echo")

    def initialize(*args, **options):
        raise ValueError(options["api_key"].get_secret_value())

    monkeypatch.setattr("mydeepagent.models.init_chat_model", initialize)
    with pytest.raises(ValueError, match="Could not initialize") as caught:
        build_model(ModelSettings(model="configured-model"))
    assert "do-not-echo" not in str(caught.value)
    assert caught.value.__suppress_context__ is True


def test_real_openai_client_constructs_without_request(monkeypatch):
    monkeypatch.setenv("MYDEEPAGENT_MODEL_API_KEY", "dummy")
    model = build_model(ModelSettings(model="configured-model", base_url="https://provider.example/v1"))
    assert model.openai_api_base == "https://provider.example/v1"
    assert model.use_responses_api is False
    assert model.store is None
    assert model.temperature is None


def test_unsupported_provider_fails_closed():
    with pytest.raises(ValueError, match="provider"):
        build_model(ModelSettings(provider="unconfigured-provider", model="configured-model"))


def test_anthropic_rejects_responses_api_mode():
    with pytest.raises(ValueError, match="only valid with the openai adapter"):
        build_model(ModelSettings(provider="anthropic", model="configured-model", use_responses_api=True))
