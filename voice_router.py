import asyncio
import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

LOGGER = logging.getLogger("ollama_proxy.voice_router")

CLASSIFIER_SIMPLE = "SIMPLE"
CLASSIFIER_COMPLEX = "COMPLEX"
QUESTION_WORDS = {
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "which",
    "whose",
    "whom",
    "did",
    "do",
    "does",
    "can",
    "could",
    "would",
    "should",
    "is",
    "are",
}


async def _noop_chunk_sink(_: str) -> None:
    return None


async def _noop_phrase_player(_: str) -> None:
    return None


async def _noop_stop_playback() -> None:
    return None


@dataclass(slots=True)
class VoiceRouterConfig:
    ollama_base_url: str
    classifier_model: str = "phi3-mini"
    response_model: str = "llama3"
    classifier_timeout_seconds: float = 1.2
    context_timeout_seconds: float = 3.5
    buffer_phrase: str = "Let me check..."


class VoiceRouter:
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        config: VoiceRouterConfig,
        get_memory: Callable[[str], Awaitable[str]],
        piper_stream_chunk: Callable[[str], Awaitable[None]] = _noop_chunk_sink,
        piper_play_phrase: Callable[[str], Awaitable[None]] = _noop_phrase_player,
        piper_stop_playback: Callable[[], Awaitable[None]] = _noop_stop_playback,
    ) -> None:
        self.http_client = http_client
        self.config = config
        self.get_memory = get_memory
        self.piper_stream_chunk = piper_stream_chunk
        self.piper_play_phrase = piper_play_phrase
        self.piper_stop_playback = piper_stop_playback

    def _classifier_fallback(self, text: str) -> str:
        stripped = text.strip().lower()
        words = [word for word in re.split(r"\s+", stripped) if word]
        if len(words) <= 3:
            return CLASSIFIER_SIMPLE
        if "?" in stripped:
            return CLASSIFIER_COMPLEX
        if any(word in QUESTION_WORDS for word in words):
            return CLASSIFIER_COMPLEX
        return CLASSIFIER_SIMPLE

    async def quick_classify(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            LOGGER.info("Classification decision: SIMPLE (empty input)")
            return CLASSIFIER_SIMPLE

        prompt = (
            "Classify the user utterance for voice-routing. "
            "Reply with exactly one word: SIMPLE or COMPLEX.\n\n"
            "SIMPLE: greetings, acknowledgements, or short chit-chat that needs no retrieval.\n"
            "COMPLEX: requests that likely need facts, memory, or retrieval.\n\n"
            f"Utterance: {normalized}\n"
            "Answer:"
        )
        payload = {
            "model": self.config.classifier_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }

        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.config.classifier_timeout_seconds):
                response = await self.http_client.post(
                    f"{self.config.ollama_base_url}/api/generate",
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()
            raw = str(data.get("response", "")).strip().upper()
            token = raw.split()[0] if raw else ""
            if token not in {CLASSIFIER_SIMPLE, CLASSIFIER_COMPLEX}:
                raise ValueError(f"Invalid classifier token: {raw}")
            elapsed = (time.perf_counter() - started) * 1000
            LOGGER.info("Classification decision: %s (%.1fms)", token, elapsed)
            return token
        except Exception as exc:
            fallback = self._classifier_fallback(normalized)
            elapsed = (time.perf_counter() - started) * 1000
            LOGGER.warning(
                "Classifier failed, using fallback=%s (%.1fms): %s",
                fallback,
                elapsed,
                exc,
            )
            return fallback

    async def stream_ollama_to_piper(self, prompt: str, context: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"{context}\n\n"
                        "Use recalled context only if it helps answer the current request."
                    ),
                }
            )
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.response_model,
            "messages": messages,
            "stream": True,
        }

        request = self.http_client.build_request(
            "POST",
            f"{self.config.ollama_base_url}/api/chat",
            json=payload,
        )
        response = await self.http_client.send(request, stream=True)
        response.raise_for_status()

        chunks: list[str] = []
        try:
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.debug("Skipping malformed Ollama stream line")
                    continue

                chunk = str(item.get("message", {}).get("content") or "")
                if not chunk:
                    continue
                chunks.append(chunk)
                await self.piper_stream_chunk(chunk)
        finally:
            await response.aclose()

        return "".join(chunks).strip()

    async def play_buffer_phrase(self, text: str, stop_signal: asyncio.Event) -> None:
        if not text.strip():
            return

        phrase_task = asyncio.create_task(self.piper_play_phrase(text))
        stop_task = asyncio.create_task(stop_signal.wait())
        done, _ = await asyncio.wait(
            {phrase_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        try:
            if stop_task in done and not phrase_task.done():
                await self.piper_stop_playback()
                phrase_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await phrase_task
            elif phrase_task in done:
                await phrase_task
        finally:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task

    async def handle_voice_input(self, transcription: str) -> dict[str, Any]:
        transcription = transcription.strip()
        if not transcription:
            return {
                "classification": CLASSIFIER_SIMPLE,
                "path": "FAST",
                "response_text": "",
                "timing_ms": 0.0,
            }

        total_started = time.perf_counter()
        classification = await self.quick_classify(transcription)

        if classification == CLASSIFIER_SIMPLE:
            LOGGER.info("Routing path: FAST")
            response_text = await self.stream_ollama_to_piper(prompt=transcription, context=None)
            total_elapsed = (time.perf_counter() - total_started) * 1000
            LOGGER.info("Voice pipeline complete path=FAST total=%.1fms", total_elapsed)
            return {
                "classification": classification,
                "path": "FAST",
                "response_text": response_text,
                "timing_ms": round(total_elapsed, 2),
            }

        LOGGER.info("Routing path: DEEP")
        stop_buffer = asyncio.Event()

        buffer_task = asyncio.create_task(
            self.play_buffer_phrase(self.config.buffer_phrase, stop_buffer)
        )

        context_started = time.perf_counter()
        try:
            async with asyncio.timeout(self.config.context_timeout_seconds):
                context = await self.get_memory(transcription)
        except Exception as exc:
            LOGGER.warning("Context retrieval failed: %s", exc)
            context = ""
        context_elapsed = (time.perf_counter() - context_started) * 1000
        LOGGER.info("Context retrieval complete in %.1fms", context_elapsed)

        stop_buffer.set()

        response_text = await self.stream_ollama_to_piper(prompt=transcription, context=context or None)

        if not buffer_task.done():
            buffer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await buffer_task
        else:
            await buffer_task

        total_elapsed = (time.perf_counter() - total_started) * 1000
        LOGGER.info("Voice pipeline complete path=DEEP total=%.1fms", total_elapsed)
        return {
            "classification": classification,
            "path": "DEEP",
            "response_text": response_text,
            "timing_ms": round(total_elapsed, 2),
            "context_timing_ms": round(context_elapsed, 2),
            "used_context": bool(context),
        }


_DEFAULT_ROUTER: VoiceRouter | None = None


def configure_default_router(router: VoiceRouter) -> None:
    global _DEFAULT_ROUTER
    _DEFAULT_ROUTER = router


def _require_default_router() -> VoiceRouter:
    if _DEFAULT_ROUTER is None:
        raise RuntimeError("Default VoiceRouter is not configured")
    return _DEFAULT_ROUTER


async def quick_classify(text: str) -> str:
    return await _require_default_router().quick_classify(text)


async def stream_ollama_to_piper(prompt: str, context: str | None = None) -> str:
    return await _require_default_router().stream_ollama_to_piper(prompt=prompt, context=context)


async def play_buffer_phrase(text: str) -> None:
    stop_signal = asyncio.Event()
    await _require_default_router().play_buffer_phrase(text=text, stop_signal=stop_signal)


async def handle_voice_input(transcription: str) -> dict[str, Any]:
    return await _require_default_router().handle_voice_input(transcription=transcription)
