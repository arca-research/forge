"""
LLM primitives: SyncLLM and AsyncLLM

TODO: add openrouter

TODO: add retry / backoff

TODO: add logging of every call (to .forge)
"""

from __future__ import annotations

from openai import OpenAI, AsyncOpenAI
from typing import Literal, Optional


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
    ):
        super().__init__()
        if backend == "openai":
            try:
                self.client = OpenAI(api_key=api_key)
            except:
                raise ValueError("OpenAI API key is not set in forge/.env.")
        elif backend == "openrouter":
            raise NotImplementedError("Openrouter is not yet supported.") # | TODO
        elif backend == "local":
            if url is None:
                raise ValueError("Local LLM client url is not set in forge/config.py")
            self.client = OpenAI(base_url=url, api_key="local")
        self.model = model


    def run(self, prompt: str) -> str:
        messages = self._guard_system() + [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content.strip()


class AsyncLLM(_BaseLLM):
    """"""

    def __init__(self,
        backend: Literal["openai", "openrouter", "local"],
        model: str,
        api_key: str,
        url: Optional[str],
    ):
        super().__init__()        
        if backend == "openai":
            try:
                self.client = AsyncOpenAI(api_key=api_key)
            except:
                raise ValueError("OpenAI API key is not set in forge/.env.")
        elif backend == "openrouter":
            raise NotImplementedError("Openrouter is not yet supported.") # | TODO
        elif backend == "local":
            if url is None:
                raise ValueError("Local LLM client url is not set in forge/config.py")
            self.client = AsyncOpenAI(base_url=url, api_key="local")
        self.model = model


    async def run(self, prompt: str) -> str:
        messages = self._guard_system() + [{"role": "user", "content": prompt}]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content.strip()
