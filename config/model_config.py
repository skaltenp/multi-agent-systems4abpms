import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / ".env.ollama")


def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4")) and not model.startswith("gpt-oss")


@dataclass
class ModelConfig:
    model: str = "gpt-5.1"
    temperature: float = 0
    seed: Optional[int] = None
    reasoning_effort: Optional[str] = None
    service_tier: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ollama(self) -> bool:
        """True if model is not a known OpenAI model (assumed Ollama)."""
        return not _is_openai_model(self.model)

    def validate_model(self) -> None:
        """Check that the model is available on the target provider. Raises RuntimeError if not."""
        if self.is_ollama:
            import requests
            base_url = self.base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            ollama_url = base_url.replace("/v1", "")
            try:
                resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
                resp.raise_for_status()
            except requests.ConnectionError:
                raise RuntimeError(f"Cannot connect to Ollama at {ollama_url}. Is Ollama running?")
            available = [m["name"] for m in resp.json().get("models", [])]
            # Match with or without :latest tag
            names = {n for n in available} | {n.split(":")[0] for n in available}
            if self.model not in names:
                raise RuntimeError(
                    f"Ollama model '{self.model}' not found. Available models: {', '.join(sorted(available))}"
                )
        else:
            from openai import OpenAI, AuthenticationError, APIConnectionError
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            client = OpenAI(api_key=api_key, base_url=self.base_url)
            try:
                models_resp = client.models.list()
            except AuthenticationError:
                raise RuntimeError("OPENAI_API_KEY is invalid.")
            except APIConnectionError as e:
                raise RuntimeError(f"Cannot connect to OpenAI API: {e}")
            available = [m.id for m in models_resp.data]
            if self.model not in available:
                raise RuntimeError(
                    f"OpenAI model '{self.model}' not found. Check for typos."
                )

    def create_llm(self, **override_kwargs) -> BaseChatModel:
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
        }

        if self.is_ollama:
            kwargs["base_url"] = self.base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            kwargs["api_key"] = self.api_key or os.environ.get("OLLAMA_API_KEY", "ollama")
            # Disable httpx pool timeout — default 600s causes deadlocks when
            # tool calls (e.g. slow SQL queries) keep the LLM connection idle
            kwargs["timeout"] = None
            # Ollama supports seed via OpenAI-compatible API
            if self.seed is not None:
                kwargs["seed"] = self.seed
            # Only gpt-oss supports reasoning_effort on Ollama;
            # other models (qwen, llama, etc.) only support think: true/false
            if not self.model.startswith("gpt-oss"):
                override_kwargs.pop("reasoning_effort", None)
            override_kwargs.pop("service_tier", None)
        else:
            # OpenAI
            if self.seed is not None:
                kwargs["seed"] = self.seed
            if self.reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.reasoning_effort
            # gpt-5.4 doesn't support reasoning_effort + tools on /v1/chat/completions
            if self.model.startswith("gpt-5.4"):
                kwargs.pop("reasoning_effort", None)
                override_kwargs.pop("reasoning_effort", None)
            if self.service_tier is not None:
                kwargs["service_tier"] = self.service_tier
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url
            if self.api_key is not None:
                kwargs["api_key"] = self.api_key

        kwargs.update(self.extra_kwargs)
        kwargs.update(override_kwargs)

        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**kwargs)
