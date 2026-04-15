import asyncio
import inspect
import logging
from datetime import datetime, timezone
from typing import Any

from hindsight_client import Hindsight
LOGGER = logging.getLogger("ollama_proxy.memory")


def _is_question(text: str) -> bool:
    """Detect if text is a question (vs. a statement of fact)."""
    normalized = text.strip().lower()
    # Check for question mark
    if normalized.endswith("?"):
        return True
    # Check for question words at the start
    question_words = ("what", "why", "how", "when", "where", "who", "which", "is", "are", "can", "could", "will", "would", "do", "does", "did", "have", "has", "should", "may", "might")
    for word in question_words:
        if normalized.startswith(word + " ") or normalized.startswith(word + "'"):
            return True
    return False


class MemoryManager:
    """Wrapper around HindsightClient for retrieval and storage."""

    def __init__(self, host: str, max_memories: int = 5, timeout_seconds: float = 4.0, bank: str | None = None) -> None:
        self.host = host.rstrip("/")
        self.max_memories = max_memories
        self.timeout = timeout_seconds
        self.bank = bank        
        self.client = Hindsight(base_url=self.host, timeout=self.timeout)

    async def _call_hindsight(self, method_name: str, **kwargs: Any) -> Any:
        method = getattr(self.client, method_name)
        try:
            result = method(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        except RuntimeError as exc:
            # Some hindsight-client builds call asyncio.run() internally,
            # which fails when FastAPI already has an active loop.
            if "event loop is already running" not in str(exc).lower():
                raise
            LOGGER.debug("Retrying hindsight.%s in worker thread", method_name)
            return await asyncio.to_thread(self._call_hindsight_sync, method_name, kwargs)

    def _call_hindsight_sync(self, method_name: str, kwargs: dict[str, Any]) -> Any:
        client = Hindsight(base_url=self.host, timeout=self.timeout)
        method = getattr(client, method_name)
        result = method(**kwargs)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    async def retrieve_memories(self, query: str, budget: int | None = None) -> list[str]:
        if not query:
            return []
        budget = budget or self.max_memories
        try:
            response = await self._call_hindsight(
                "arecall",
                query=query,
                budget="low",
                bank_id=self.bank,
            )
            # response.results is a list of RecallResult objects
            results = response.results
            return [item.text for item in results if item.text]
        except Exception as e:
            LOGGER.warning(f"Hindsight recall failed: {e}")
            return []

    async def build_context_block(self, query: str) -> str:
        memories = await self.retrieve_memories(query)
        if not memories:
            return ""
        bullet_list = "\n".join(f"- {memory}" for memory in memories)
        return (
            "Relevant past interactions and home context:\n"
            f"{bullet_list}\n\n"
            "Use these memories only if they help answer the current request accurately."
        )

    async def store_interaction(self, query: str, response_text: str) -> bool:
        if not query or not response_text:
            return False
        if not self.bank:
            # No bank configured; skip retention
            return False
        if _is_question(query):
            # Skip storing questions; only store statements/facts
            return False
        memory_text = f"User: {query}\nAssistant: {response_text}"
        metadata = {
            "source": "ollama-proxy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "conversation"
        }
        try:
            # Use a longer timeout for retention since it's best-effort
            client = Hindsight(base_url=self.host, timeout=10.0)
            await client.aretain(
                bank_id=self.bank,
                content=memory_text,
                context=str(metadata),
            )
            return True
        except TimeoutError:
            # Storage is best-effort and should never surface as a hard failure.
            LOGGER.debug("Hindsight retain timed out (best-effort operation)")
            return False
        except Exception as e:
            LOGGER.debug(f"Hindsight retain failed: {e}")
            return False
