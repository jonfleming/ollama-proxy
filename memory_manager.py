import logging
from datetime import datetime, timezone
from typing import Any

from hindsight_client import Hindsight
LOGGER = logging.getLogger("ollama_proxy.memory")


class MemoryManager:
    """Wrapper around HindsightClient for retrieval and storage."""

    def __init__(self, host: str, max_memories: int = 5, timeout_seconds: float = 4.0, bank: str | None = None) -> None:
        self.host = host.rstrip("/")
        self.max_memories = max_memories
        self.timeout = timeout_seconds
        self.bank = bank        
        self.client = Hindsight(base_url=self.host, timeout=self.timeout)

    async def retrieve_memories(self, query: str, limit: int | None = None) -> list[str]:
        if not query:
            return []
        limit = limit or self.max_memories
        try:
            results, error = await self.client.recall(query=query, limit=limit, bank_id=self.bank)
            # Each result is a dict with at least a 'text' field
            if error:
                LOGGER.warning(f"Hindsight search returned error: {error}")
                return []
            return [item.get("text", "") for item in results if item.get("text")]
        except Exception as e:
            LOGGER.warning(f"Hindsight search failed: {e}")
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
        memory_text = f"User: {query}\nAssistant: {response_text}"
        metadata = {
            "source": "ollama-proxy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "conversation"
        }
        try:
            await self.client.retain(content=memory_text, context=str(metadata), bank_id=self.bank)
            return True
        except Exception as e:
            LOGGER.warning(f"Hindsight add failed: {e}")
            return False
