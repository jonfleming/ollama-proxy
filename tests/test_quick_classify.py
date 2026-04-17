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
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)
        return self._resp


class TimeoutClient:
    def __init__(self):
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        raise TimeoutError()


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


@pytest.mark.asyncio
async def test_quick_classify_short_circuits_non_personal_with_heuristic():
    resp = MockResponse({"response": "COMPLEX_PERSONAL"})
    client = DummyClient(resp)
    cfg = VoiceRouterConfig(
        ollama_base_url="http://localhost:11434",
        classifier_only_on_personal_hint=True,
    )
    router = VoiceRouter(http_client=client, config=cfg, get_memory=lambda q: "")

    token = await router.quick_classify("What time is it?")
    assert token == CLASSIFIER_COMPLEX_GENERAL
    assert client.calls == 0


@pytest.mark.asyncio
async def test_quick_classify_uses_breaker_after_timeout_streak():
    client = TimeoutClient()
    cfg = VoiceRouterConfig(
        ollama_base_url="http://localhost:11434",
        classifier_only_on_personal_hint=False,
        classifier_timeout_seconds=0.05,
        classifier_timeout_streak_breaker=2,
        classifier_breaker_cooldown_seconds=60.0,
    )
    router = VoiceRouter(http_client=client, config=cfg, get_memory=lambda q: "")

    # Personal marker forces classifier attempt when short-circuit mode is off.
    q = "Can you remind me what I said about my project?"
    t1 = await router.quick_classify(q)
    t2 = await router.quick_classify(q)
    t3 = await router.quick_classify(q)

    assert t1 == CLASSIFIER_COMPLEX_PERSONAL
    assert t2 == CLASSIFIER_COMPLEX_PERSONAL
    assert t3 == CLASSIFIER_COMPLEX_PERSONAL
    # Third call should skip HTTP because breaker is open.
    assert client.calls == 2
