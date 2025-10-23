"""
LLM primitives: SyncLLM and AsyncLLM

TODO: add openrouter

TODO: add logging of every call (to .forge)
"""

from __future__ import annotations

from openai import OpenAI, AsyncOpenAI
from typing import Literal, Optional
from pathlib import Path
import time, asyncio

from ..config import log

class _BaseLLM:
    def __init__(self, system: Optional[list[dict]] = None):
        self.system = system or []

    def set_system(self, content: str) -> None:
        self.system = [{"role": "system", "content": content}]

    def _guard_system(self) -> list[dict]:
        if not self.system:
            raise ValueError("System prompt is not set.")
        return self.system


class SyncLLM(_BaseLLM):
    """"""

    def __init__(self,
        backend: Literal["openai", "openrouter", "local"],
        model: str,
        api_key: str,
        url: Optional[str],
        log_path: Optional[Path] = None, # | TODO
        retries: int = 0,
    ):
        super().__init__()
        self.backend = backend
        if self.backend == "openai":
            try:
                self.client = OpenAI(api_key=api_key)
            except:
                raise ValueError("OpenAI API key is not set in forge/.env.")
        elif self.backend == "openrouter":
            raise NotImplementedError("Openrouter is not yet supported.") # | TODO
        elif self.backend == "local":
            if url is None:
                raise ValueError("Local LLM client url is not set in forge/config.py")
            self.client = OpenAI(base_url=url, api_key="local")
        self.model = model
        self.log_path = log_path
        self.retries = retries


    def run(self, prompt: str) -> str:
        messages = self._guard_system() + [{"role": "user", "content": prompt}]
        for attempt in range(self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                output = response.choices[0].message.content.strip()
                return output
            except Exception as exc:
                if attempt < self.retries - 1:
                    log.warning("Encountered SyncLLM exception during attempt %s: [%s]: %s",
                        attempt, type(exc).__name__, exc)
                    wait_time = min(2 ** attempt, 30)
                    time.sleep(wait_time)
                else:
                    raise exc


class AsyncLLM(_BaseLLM):
    """"""

    def __init__(self,
        backend: Literal["openai", "openrouter", "local"],
        model: str,
        api_key: str,
        url: Optional[str],
        log_path: Optional[Path] = None, # | TODO
        retries: int = 0,
    ):
        super().__init__()
        self.backend = backend
        if self.backend == "openai":
            try:
                self.client = AsyncOpenAI(api_key=api_key)
            except:
                raise ValueError("OpenAI API key is not set in forge/.env.")
        elif self.backend == "openrouter":
            raise NotImplementedError("Openrouter is not yet supported.") # | TODO
        elif self.backend == "local":
            if url is None:
                raise ValueError("Local LLM client url is not set in forge/config.py")
            self.client = AsyncOpenAI(base_url=url, api_key="local")
        self.model = model
        self.log_path = log_path
        self.retries = retries


    async def run(self, prompt: str) -> str:
        messages = self._guard_system() + [{"role": "user", "content": prompt}]
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                output = response.choices[0].message.content.strip()
                return output
            except Exception as exc:
                if attempt < self.retries - 1:
                    log.warning("Encountered AsyncLLM exception during attempt %s: [%s]: %s",
                        attempt, type(exc).__name__, exc)
                    wait_time = min(2 ** attempt, 30)
                    await asyncio.sleep(wait_time)
                else:
                    raise exc
