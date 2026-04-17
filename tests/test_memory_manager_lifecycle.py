import pytest

import memory_manager as mm


class FakeHindsight:
    instances = []

    def __init__(self, base_url: str, timeout: float = 300.0, api_key=None):
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key
        self.closed = False
        FakeHindsight.instances.append(self)

    async def aretain(self, **kwargs):
        return kwargs

    async def aclose(self):
        self.closed = True

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_memory_manager_aclose_closes_primary_client(monkeypatch):
    FakeHindsight.instances.clear()
    monkeypatch.setattr(mm, "Hindsight", FakeHindsight)

    manager = mm.MemoryManager(host="http://hindsight.local", bank="test-bank")
    assert len(FakeHindsight.instances) == 1

    await manager.aclose()

    assert FakeHindsight.instances[0].closed is True


@pytest.mark.asyncio
async def test_store_interaction_closes_temporary_client(monkeypatch):
    FakeHindsight.instances.clear()
    monkeypatch.setattr(mm, "Hindsight", FakeHindsight)

    manager = mm.MemoryManager(host="http://hindsight.local", bank="test-bank")
    result = await manager.store_interaction(
        query="I prefer almond milk in coffee",
        response_text="Got it.",
    )

    assert result is True
    assert len(FakeHindsight.instances) == 2
    assert FakeHindsight.instances[1].closed is True
