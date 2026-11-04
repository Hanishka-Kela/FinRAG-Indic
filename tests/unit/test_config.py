from finrag_indic.config import Settings


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.api_port == 8000
    assert settings.llm_provider == "mock"
    assert settings.llm_model is None


def test_settings_load_prefixed_environment(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FINRAG_ENVIRONMENT", "test")
    monkeypatch.setenv("FINRAG_API_PORT", "8123")
    monkeypatch.setenv("FINRAG_LLM_MODEL", "pilot-model")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.api_port == 8123
    assert settings.llm_model == "pilot-model"


def test_empty_optional_provider_values_become_none(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FINRAG_LLM_BASE_URL", "")
    monkeypatch.setenv("FINRAG_LLM_API_KEY", "")

    settings = Settings(_env_file=None)

    assert settings.llm_base_url is None
    assert settings.llm_api_key is None

