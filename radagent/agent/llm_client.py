"""Minimal client for any OpenAI-compatible chat-completions server.

Works with open models served locally - Ollama (``http://localhost:11434/v1``),
vLLM, LM Studio, llama.cpp server - as the AIMA project asks for an *open* planner.
Only ``requests`` is needed.
"""
from __future__ import annotations

import json
import os
import re

import requests


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """Return the first JSON object found in ``text`` (tolerates code fences and prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise LLMError(f"no JSON object in model output: {text[:200]!r}")


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "LLM_API_KEY", timeout_s: float = 60,
                 temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env, "ollama")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self._json_mode = True

    def chat(self, messages: list[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "stream": False}
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout_s)
            if r.status_code == 400 and self._json_mode:          # server without JSON mode
                self._json_mode = False
                payload.pop("response_format", None)
                r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout_s)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            raise LLMError(str(e)) from e

    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=5,
                             headers={"Authorization": f"Bearer {self.api_key}"})
            return r.ok
        except requests.RequestException:
            return False
