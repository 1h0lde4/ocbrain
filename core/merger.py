"""
core/merger.py — V2.1: smart merge avoids extra LLM call for ≤2 modules.
Strategy:
  - 0 results   → error message
  - 1 result    → pass-through (fastest)
  - 2 results   → template join (no LLM, ~0ms)
  - 3+ results  → LLM weave only when content truly conflicts
"""
import logging
from .dispatcher import TaskResult
from .config import config

log = logging.getLogger(__name__)

_JOIN_TEMPLATE = "{first}\n\nAdditionally: {second}"


async def merge(results: list[TaskResult], original_query: str) -> str:
    if not results:
        return "I was unable to process your request."

    valid  = [r for r in results if r.result.source != "error"]
    errors = [r for r in results if r.result.source == "error"]

    if not valid:
        return "\n".join(r.result.answer for r in errors)

    # ── 1 module — direct pass-through ───────────────────────
    if len(valid) == 1:
        ans = valid[0].result.answer
        if errors:
            ans += "\n\n" + "\n".join(r.result.answer for r in errors)
        return ans

    # ── 2 modules — fast template join (no LLM call) ─────────
    if len(valid) == 2:
        a1 = valid[0].result.answer.strip()
        a2 = valid[1].result.answer.strip()
        # If answers are near-identical, just return the first
        if _word_overlap(a1, a2) > 0.80:
            return a1
        # If one is very short (supplement), append naturally
        if len(a2.split()) < 40:
            return f"{a1}\n\n{a2}"
        if len(a1.split()) < 40:
            return f"{a2}\n\n{a1}"
        # Standard template join
        merged = _JOIN_TEMPLATE.format(first=a1, second=a2)
        if errors:
            merged += "\n\n" + "\n".join(r.result.answer for r in errors)
        return merged

    # ── 3+ modules — deduplicate then LLM weave ──────────────
    unique = _deduplicate([r.result.answer for r in valid])
    if len(unique) == 1:
        return unique[0]

    # Check if answers actually conflict before paying LLM cost
    if _answers_compatible(unique):
        return "\n\n".join(unique)

    return await _weave(unique, original_query)


def _deduplicate(answers: list[str]) -> list[str]:
    unique = []
    for ans in answers:
        if not any(_word_overlap(ans, u) > 0.85 for u in unique):
            unique.append(ans)
    return unique


def _word_overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def _answers_compatible(answers: list[str]) -> bool:
    """True if answers don't contain explicit contradictions."""
    contradiction_pairs = [
        ("yes", "no"), ("true", "false"), ("correct", "incorrect"),
        ("does", "does not"), ("can", "cannot"), ("is", "is not"),
    ]
    combined = " ".join(a.lower() for a in answers)
    for pos, neg in contradiction_pairs:
        if pos in combined and neg in combined:
            return False
    return True


async def _weave(answers: list[str], query: str) -> str:
    import httpx
    host  = config.get("global.ollama_host") or "http://localhost:11434"
    model = config.get("global.classifier_model") or "mistral"
    parts = "\n\n---\n\n".join(f"[{i+1}]:\n{a}" for i, a in enumerate(answers))
    prompt = (
        f"Combine these answers into one clear, non-repetitive response.\n"
        f"Query: {query}\n\nAnswers:\n{parts}\n\nUnified response:"
    )
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            return resp.json().get("response", "").strip()
    except Exception:
        return "\n\n".join(answers)
