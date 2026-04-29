"""
modules/coding/module.py — Coding expert module.
Handles code generation, debugging, and explanation tasks.
"""
import ast
import time
import httpx
from pathlib import Path

from modules.base import BaseModule, ModuleResult
from core.config import config


class Module(BaseModule):
    name = "coding"

    async def run(self, task: str, context) -> ModuleResult:
        t0     = time.monotonic()
        chunks = self.retrieve(task, k=5)
        prompt = self._build_prompt(task, chunks, context)
        answer = await self._call_external_raw(prompt)
        answer = self._validate_code(answer)
        self.save_training_pair(task, answer)
        return ModuleResult(
            answer=answer,
            source="external",
            chunks_used=chunks,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    async def run_own(self, task: str, context) -> ModuleResult:
        t0     = time.monotonic()
        chunks = self.retrieve(task, k=5)
        prompt = self._build_prompt(task, chunks, context)
        answer = await self._call_own_raw(prompt)
        answer = self._validate_code(answer)
        return ModuleResult(
            answer=answer,
            source="native",
            chunks_used=chunks,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    def _build_prompt(self, task: str, chunks: list, context) -> str:
        ctx_str  = context.format_for_prompt(3) if context else ""
        kb_str   = "\n\n".join(chunks) if chunks else "No relevant code examples found."
        lang     = ""
        if context:
            langs = context.get_entity("languages", 1)
            if langs:
                lang = f"Use {langs[0]}. "
        return (
            f"You are an expert software engineer.\n"
            f"{ctx_str}\n\n"
            f"Relevant code examples:\n{kb_str}\n\n"
            f"{lang}Task: {task}\n\n"
            f"Provide working, well-commented code:"
        )

    def _validate_code(self, response: str) -> str:
        """Try to validate Python syntax if response contains a Python block."""
        import re
        py_block = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if py_block:
            code = py_block.group(1)
            try:
                ast.parse(code)
            except SyntaxError as e:
                response += f"\n\n⚠️ Syntax note: {e}"
        return response

    async def _call_external_raw(self, prompt: str) -> str:
        state = config.get_module_state(self.name)
        model = state.get("bootstrap_model", "mistral")
        host  = config.get("global.ollama_host") or "http://localhost:11434"
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{host}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                return resp.json().get("response", "").strip()
        except Exception as e:
            return f"[Coding module error: {e}]"

    async def _call_own_raw(self, prompt: str) -> str:
        state = config.get_module_state(self.name)
        model = state.get("own_model_tag") or state.get("bootstrap_model", "mistral")
        host  = config.get("global.ollama_host") or "http://localhost:11434"
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{host}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                return resp.json().get("response", "").strip()
        except Exception as e:
            return f"[Coding own-model error: {e}]"
