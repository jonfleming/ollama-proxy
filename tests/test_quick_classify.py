import asyncio
import re

import pytest

from voice_router import (
    VoiceRouter,
    VoiceRouterConfig,
    CLASSIFIER_SIMPLE,
    CLASSIFIER_COMPLEX_PERSONAL,
    CLASSIFIER_COMPLEX_GENERAL,
)


class MockResponse:
    def __init__(self, json_data, status_code=200, text=None):
        self._json = json_data
        self.status_code = status_code
        self.text = text or str(json_data)

    def json(self):
        return self._json

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")


class DummyClient:
    def __init__(self, response: MockResponse):
        self._resp = response

    async def post(self, *args, **kwargs):
        await asyncio.sleep(0)
        return self._resp


@pytest.mark.asyncio
async def test_quick_classify_accepts_noisy_complex():
    # classifier returns messy casing and extra metadata
    resp = MockResponse({
        "model": "llama3.2:1b",
        "response": "COMplex",
        "done": True,
    })
    client = DummyClient(resp)
    cfg = VoiceRouterConfig(ollama_base_url="http://localhost:11434")
    router = VoiceRouter(http_client=client, config=cfg, get_memory=lambda q: "")

    token = await router.quick_classify("Is the meeting tomorrow?")
    assert token == CLASSIFIER_COMPLEX_GENERAL


@pytest.mark.asyncio
async def test_quick_classify_personal_token():
    resp = MockResponse({"response": "COMPLEX_PERSONAL"})
    client = DummyClient(resp)
    cfg = VoiceRouterConfig(ollama_base_url="http://localhost:11434")
    router = VoiceRouter(http_client=client, config=cfg, get_memory=lambda q: "")

    token = await router.quick_classify("What did I say about my project?")
    assert token == CLASSIFIER_COMPLEX_PERSONAL


def test_classifier_fallback_personal_marker():
    cfg = VoiceRouterConfig(ollama_base_url="http://localhost:11434")
    router = VoiceRouter(http_client=None, config=cfg, get_memory=lambda q: "")

    # short personal utterance
    f = router._classifier_fallback("Tell me about my calendar")
    assert f == CLASSIFIER_COMPLEX_PERSONAL

    # general question
    g = router._classifier_fallback("What is the capital of France?")
    assert g == CLASSIFIER_COMPLEX_GENERAL
