"""
core/orchestrator.py — The main brain coordinator.
Loaded once at startup; handles every user query end-to-end.
"""
import asyncio
from typing import Optional

from . import classifier, decomposer, dispatcher, merger, parser
from .config import config
from .context import ContextMemory
from .model_router import ModelRouter


class Orchestrator:
    def __init__(self, modules: dict, context: ContextMemory, router: ModelRouter):
        self.modules = modules
        self.context = context
        self.router  = router

    async def handle(self, query: str) -> str:
        try:
            # 1. Parse
            parsed = parser.parse(query)

            # 2. Classify
            labels = await classifier.label(parsed, self.context)

            # 3. Decompose into task DAG
            tasks = decomposer.build(parsed, labels)

            # 4. Dispatch (parallel/serial)
            results = await dispatcher.run(tasks, self.router, self.context, self.modules)

            # 5. Merge answers
            answer = await merger.merge(results, query)

            # 6. Save to context memory
            modules_used = [r.module for r in results]
            entities = {
                "urls":      parsed.entities.get("urls", []),
                "languages": parsed.entities.get("languages", []),
                "filenames": parsed.entities.get("filenames", []),
            }
            self.context.save(query, modules_used, answer, entities)

            return answer

        except Exception as e:
            import logging
            logging.getLogger("ocbrain").error(
                f"orchestrator.handle() error: {type(e).__name__}: {e}", exc_info=True
            )
            return (
                f"Sorry, I encountered an error processing your request.\n"
                f"Error: {type(e).__name__}: {e}\n\n"
                f"Check the terminal running OCBrain for the full traceback."
            )

    def status(self) -> dict:
        return {
            name: mod.health()
            for name, mod in self.modules.items()
        }
