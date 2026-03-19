"""Serializes LangChain response messages to JSON using built-in dumpd for lossless output."""

import json
import os

from langchain_core.load import dumpd
from langchain_core.messages import AIMessage


def compute_token_summary(messages) -> dict[str, int]:
    """Aggregate token usage across all AIMessages in a response."""
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            input_tokens += m.usage_metadata["input_tokens"]
            output_tokens += m.usage_metadata["output_tokens"]
            total_tokens += m.usage_metadata["total_tokens"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def save_messages_json(messages, session_dir: str, filename: str) -> None:
    """Serialize LangChain messages and write to a JSON file in session_dir."""
    os.makedirs(session_dir, exist_ok=True)
    data = {
        "messages": [dumpd(m) for m in messages],
        "token_usage": compute_token_summary(messages),
    }
    path = os.path.join(session_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
