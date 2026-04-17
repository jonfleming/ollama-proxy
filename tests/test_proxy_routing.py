from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

import main


class FakeRouter:
    def __init__(self, classification: str):
        self.classification = classification
        self.config = SimpleNamespace(context_timeout_seconds=0.1)

    async def quick_classify(self, text: str) -> str:
        return self.classification


class FakeMemoryManager:
    def __init__(self):
        self.context_calls = 0

    async def build_context_block(self, query: str) -> str:
        self.context_calls += 1
        return "Relevant past interactions and home context:\n- User likes tea"

    async def store_interaction(self, query: str, response_text: str) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class FakeHttpClient:
    def __init__(self):
        self.last_payload = None

    async def post(self, url: str, json: dict):
        self.last_payload = json
        return httpx.Response(
            status_code=200,
            json={"message": {"content": "ok"}},
            headers={"content-type": "application/json"},
        )

    async def aclose(self) -> None:
        return None


def test_chat_proxy_uses_context_for_complex_personal():
    with TestClient(main.app) as client:
        fake_memory = FakeMemoryManager()
        fake_http = FakeHttpClient()
        main.app.state.memory_manager = fake_memory
        main.app.state.http = fake_http
        main.app.state.voice_router = FakeRouter("COMPLEX_PERSONAL")

        response = client.post(
            "/api/chat",
            json={
                "model": "llama3",
                "stream": False,
                "messages": [{"role": "user", "content": "What did I say about my calendar?"}],
            },
        )

        assert response.status_code == 200
        assert fake_memory.context_calls == 1
        assert fake_http.last_payload["messages"][0]["role"] == "system"


def test_chat_proxy_skips_context_for_complex_general():
    with TestClient(main.app) as client:
        fake_memory = FakeMemoryManager()
        fake_http = FakeHttpClient()
        main.app.state.memory_manager = fake_memory
        main.app.state.http = fake_http
        main.app.state.voice_router = FakeRouter("COMPLEX_GENERAL")

        response = client.post(
            "/api/chat",
            json={
                "model": "llama3",
                "stream": False,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
            },
        )

        assert response.status_code == 200
        assert fake_memory.context_calls == 0
        assert fake_http.last_payload["messages"][0]["role"] == "user"
