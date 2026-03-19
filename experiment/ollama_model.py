"""Ollama model warmup/unload helper. Called from experiment shell scripts."""

import argparse
import os
import sys

import requests

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").removesuffix("/v1")


def get_loaded_models(base_url: str) -> list[str]:
    resp = requests.get(f"{base_url}/api/ps")
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def unload_model(base_url: str, model: str) -> None:
    requests.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": [], "keep_alive": 0, "stream": False},
    ).raise_for_status()


def load_model(base_url: str, model: str) -> None:
    requests.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": [], "keep_alive": -1, "stream": False},
    ).raise_for_status()


def warmup(base_url: str, model: str) -> None:
    loaded = get_loaded_models(base_url)
    if loaded:
        print(f"  Unloading currently loaded models: {loaded}")
        for m in loaded:
            unload_model(base_url, m)
            print(f"    Unloaded {m}")

    print(f"  Warming up Ollama model '{model}' (keep_alive=-1)...")
    load_model(base_url, model)
    print("  Model loaded and pinned in memory.")


def unload(base_url: str, model: str) -> None:
    print(f"  Unloading Ollama model '{model}' (keep_alive=0)...")
    unload_model(base_url, model)
    print("  Model unloaded.")


def unload_all(base_url: str) -> None:
    loaded = get_loaded_models(base_url)
    if not loaded:
        print("  No models currently loaded.")
        return
    print(f"  Unloading all loaded models: {loaded}")
    for m in loaded:
        unload_model(base_url, m)
        print(f"    Unloaded {m}")


def main():
    parser = argparse.ArgumentParser(description="Ollama model warmup/unload helper")
    parser.add_argument("action", choices=["warmup", "unload", "unload_all"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    try:
        if args.action == "unload_all":
            unload_all(args.base_url)
        elif args.model is None:
            parser.error("--model is required for warmup/unload")
        elif args.action == "warmup":
            warmup(args.base_url, args.model)
        else:
            unload(args.base_url, args.model)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to Ollama at {args.base_url}. Is Ollama running?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
