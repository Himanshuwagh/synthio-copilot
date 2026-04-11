# config.py — Load LLM settings from config.ini and .env (API keys).

from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
_DEFAULT_INI = _ROOT / "config.ini"
_EXAMPLE_INI = _ROOT / "config.example.ini"


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    service_role_key: str


@dataclass(frozen=True)
class LLMSettings:
    provider: str  # ollama | openai | anthropic
    model: str
    temperature: float
    ollama_host: str | None
    openai_api_key: str | None
    anthropic_api_key: str | None


def _ini_path() -> Path:
    p = os.environ.get("SYNTHIO_CONFIG")
    if p:
        return Path(p).expanduser().resolve()
    if _DEFAULT_INI.is_file():
        return _DEFAULT_INI
    if _EXAMPLE_INI.is_file():
        return _EXAMPLE_INI
    return _DEFAULT_INI


def load_llm_settings() -> LLMSettings:
    # Prefer values from project .env over inherited shell env (common gotcha:
    # empty ANTHROPIC_API_KEY or LLM_PROVIDER in the shell would otherwise win).
    load_dotenv(_ROOT / ".env", override=True)

    path = _ini_path()
    parser = ConfigParser()
    if path.is_file():
        parser.read(path, encoding="utf-8")

    def get(section: str, key: str, fallback: str = "") -> str:
        if parser.has_option(section, key):
            v = parser.get(section, key, fallback=fallback)
            return v.strip() if isinstance(v, str) else str(v)
        return fallback

    provider = (
        os.environ.get("LLM_PROVIDER")
        or get("llm", "provider", "ollama")
    ).strip().lower()

    model = (
        os.environ.get("LLM_MODEL")
        or get("llm", "model", "llama3.1:8b")
    ).strip()

    temp_raw = os.environ.get("LLM_TEMPERATURE") or get("llm", "temperature", "0")
    try:
        temperature = float(temp_raw)
    except ValueError:
        temperature = 0.0

    ollama_host = os.environ.get("OLLAMA_HOST") or None
    if not ollama_host:
        host = get("ollama", "host", "").strip()
        ollama_host = host or None

    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None

    if provider == "openai" and not openai_api_key:
        raise RuntimeError(
            "LLM provider is 'openai' but OPENAI_API_KEY is not set. "
            "Add it to .env or switch provider to ollama in config.ini."
        )
    if provider == "anthropic" and not anthropic_api_key:
        raise RuntimeError(
            "LLM provider is 'anthropic' but ANTHROPIC_API_KEY is not set. "
            "Add it to .env or switch provider to ollama in config.ini."
        )

    if provider not in ("ollama", "openai", "anthropic"):
        raise RuntimeError(
            f"Unknown LLM provider '{provider}'. Use ollama, openai, or anthropic."
        )

    return LLMSettings(
        provider=provider,
        model=model,
        temperature=temperature,
        ollama_host=ollama_host,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
    )


_settings: LLMSettings | None = None


def get_llm_settings() -> LLMSettings:
    global _settings
    if _settings is None:
        _settings = load_llm_settings()
    return _settings


def reset_llm_settings_cache() -> None:
    """For tests: force reload on next get_llm_settings()."""
    global _settings
    _settings = None


def get_supabase_settings() -> SupabaseSettings | None:
    """Return Supabase credentials when SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set."""
    load_dotenv(_ROOT / ".env", override=True)
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        return None
    return SupabaseSettings(url=url, service_role_key=key)
