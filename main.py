import asyncio
import copy
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from memory_manager import MemoryManager
from voice_router import VoiceRouter, VoiceRouterConfig, configure_default_router

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("ollama_proxy")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://100.120.84.114:11434").rstrip("/")
HINDSIGHT_HOST = os.getenv("HINDSIGHT_HOST", "http://100.111.132.40:8888").rstrip("/")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))
HINDSIGHT_MAX_MEMORIES = int(os.getenv("HINDSIGHT_MAX_MEMORIES", "5"))
HINDSIGHT_BANK = os.getenv("HINDSIGHT_BANK", "amicus-2026")
VOICE_CLASSIFIER_MODEL = os.getenv("VOICE_CLASSIFIER_MODEL", "llama3.2:1b")
VOICE_RESPONSE_MODEL = os.getenv("VOICE_RESPONSE_MODEL", "llama3")
VOICE_CLASSIFIER_TIMEOUT_SECONDS = float(os.getenv("VOICE_CLASSIFIER_TIMEOUT_SECONDS", "1.2"))
VOICE_CLASSIFIER_ONLY_ON_PERSONAL_HINT = (
    os.getenv("VOICE_CLASSIFIER_ONLY_ON_PERSONAL_HINT", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)
VOICE_CLASSIFIER_TIMEOUT_STREAK_BREAKER = int(
    os.getenv("VOICE_CLASSIFIER_TIMEOUT_STREAK_BREAKER", "2")
)
VOICE_CLASSIFIER_BREAKER_COOLDOWN_SECONDS = float(
    os.getenv("VOICE_CLASSIFIER_BREAKER_COOLDOWN_SECONDS", "30.0")
)
VOICE_CONTEXT_TIMEOUT_SECONDS = float(os.getenv("VOICE_CONTEXT_TIMEOUT_SECONDS", "3.5"))
VOICE_BUFFER_PHRASE = os.getenv("VOICE_BUFFER_PHRASE", "Let me check...")
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def piper_stream_chunk(chunk: str) -> None:
        # Hook this to your Piper async streaming sink.
        LOGGER.debug("Piper chunk: %s", chunk)

    async def piper_play_phrase(text: str) -> None:
        # Hook this to your Piper one-shot/non-blocking phrase playback.
        LOGGER.info("Piper buffer phrase: %s", text)

    async def piper_stop_playback() -> None:
        # Hook this to Piper cancellation if supported.
        LOGGER.debug("Piper stop playback requested")

    async def get_memory_context(query: str) -> str:
        return await app.state.memory_manager.build_context_block(query)

    app.state.http = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    app.state.memory_manager = MemoryManager(
        host=HINDSIGHT_HOST,
        bank=HINDSIGHT_BANK,
        max_memories=HINDSIGHT_MAX_MEMORIES,
    )
    app.state.voice_router = VoiceRouter(
        http_client=app.state.http,
        config=VoiceRouterConfig(
            ollama_base_url=OLLAMA_BASE_URL,
            classifier_model=VOICE_CLASSIFIER_MODEL,
            response_model=VOICE_RESPONSE_MODEL,
            classifier_timeout_seconds=VOICE_CLASSIFIER_TIMEOUT_SECONDS,
            classifier_only_on_personal_hint=VOICE_CLASSIFIER_ONLY_ON_PERSONAL_HINT,
            classifier_timeout_streak_breaker=VOICE_CLASSIFIER_TIMEOUT_STREAK_BREAKER,
            classifier_breaker_cooldown_seconds=VOICE_CLASSIFIER_BREAKER_COOLDOWN_SECONDS,
            context_timeout_seconds=VOICE_CONTEXT_TIMEOUT_SECONDS,
            buffer_phrase=VOICE_BUFFER_PHRASE,
        ),
        get_memory=get_memory_context,
        piper_stream_chunk=piper_stream_chunk,
        piper_play_phrase=piper_play_phrase,
        piper_stop_playback=piper_stop_playback,
    )
    configure_default_router(app.state.voice_router)
    LOGGER.info("Proxy starting with Ollama at %s", OLLAMA_BASE_URL)
    try:
        yield
    finally:
        await app.state.memory_manager.aclose()
        await app.state.http.aclose()


app = FastAPI(
    title="ollama-hindsight-proxy",
    version="0.1.0",
    lifespan=lifespan,
)


def _normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(part.strip() for part in parts if part).strip()
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or value).strip()
    return str(value).strip()


def extract_user_query(payload: dict[str, Any], endpoint: str) -> str:
    if endpoint == "generate":
        return _normalize_content(payload.get("prompt"))

    messages = payload.get("messages", [])
    for message in reversed(messages):
        if message.get("role") == "user":
            return _normalize_content(message.get("content"))
    return ""


def inject_context(payload: dict[str, Any], endpoint: str, context_block: str) -> dict[str, Any]:
    updated_payload = copy.deepcopy(payload)

    if endpoint == "generate":
        original_prompt = _normalize_content(updated_payload.get("prompt"))
        updated_payload["prompt"] = (
            f"{context_block}\n\n"
            f"Current request:\n{original_prompt}"
        )
        return updated_payload

    system_message = {
        "role": "system",
        "content": (
            f"{context_block}\n\n"
            "Use the recalled context only when it is relevant. "
            "Do not mention hidden memory retrieval unless the user asks about it."
        ),
    }

    messages = list(updated_payload.get("messages", []))
    if messages and messages[0].get("role") == "system":
        existing = _normalize_content(messages[0].get("content"))
        messages[0]["content"] = f"{existing}\n\n{system_message['content']}".strip()
    else:
        messages.insert(0, system_message)

    updated_payload["messages"] = messages
    return updated_payload


def extract_assistant_text(payload: dict[str, Any], endpoint: str) -> str:
    if endpoint == "generate":
        return _normalize_content(payload.get("response"))
    return _normalize_content(payload.get("message", {}).get("content"))


async def store_interaction_if_possible(request: Request, query: str, response_text: str) -> None:
    if not query or not response_text:
        return
    memory_manager: MemoryManager = request.app.state.memory_manager
    await memory_manager.store_interaction(query=query, response_text=response_text)


async def passthrough_get(request: Request, upstream_path: str) -> Response:
    http_client: httpx.AsyncClient = request.app.state.http
    upstream_url = f"{OLLAMA_BASE_URL}{upstream_path}"

    try:
        upstream_response = await http_client.get(upstream_url)
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
        )
    except httpx.HTTPError as exc:
        LOGGER.exception("Failed GET passthrough to %s", upstream_url)
        return JSONResponse(
            status_code=502,
            content={"error": "Unable to reach Ollama", "details": str(exc)},
        )


async def passthrough_post(request: Request, upstream_path: str) -> Response:
    http_client: httpx.AsyncClient = request.app.state.http
    upstream_url = f"{OLLAMA_BASE_URL}{upstream_path}"

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}

    try:
        upstream_response = await http_client.post(upstream_url, json=payload)
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
        )
    except httpx.HTTPError as exc:
        LOGGER.exception("Failed POST passthrough to %s", upstream_url)
        return JSONResponse(
            status_code=502,
            content={"error": "Unable to reach Ollama", "details": str(exc)},
        )


async def passthrough_post_streaming(request: Request, upstream_path: str) -> Response:
    http_client: httpx.AsyncClient = request.app.state.http
    upstream_url = f"{OLLAMA_BASE_URL}{upstream_path}"

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}

    try:
        upstream_request = http_client.build_request("POST", upstream_url, json=payload)
        upstream_response = await http_client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        LOGGER.exception("Failed streaming POST passthrough to %s", upstream_url)
        return JSONResponse(
            status_code=502,
            content={"error": "Unable to reach Ollama", "details": str(exc)},
        )

    if upstream_response.status_code >= 400:
        error_body = await upstream_response.aread()
        await upstream_response.aclose()
        return Response(
            content=error_body,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
        )

    async def raw_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_raw():
                if chunk:
                    yield chunk
        finally:
            await upstream_response.aclose()

    return StreamingResponse(
        raw_stream(),
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get("content-type", "application/x-ndjson"),
    )


async def proxy_generation_request(request: Request, endpoint: str) -> Response:
    http_client: httpx.AsyncClient = request.app.state.http
    memory_manager: MemoryManager = request.app.state.memory_manager
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    request_started = time.perf_counter()

    parse_started = time.perf_counter()
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload"})
    parse_elapsed = (time.perf_counter() - parse_started) * 1000

    stream = bool(payload.get("stream", False))
    LOGGER.info(
        "Proxy request id=%s endpoint=%s stream=%s parse=%.1fms",
        request_id,
        endpoint,
        stream,
        parse_elapsed,
    )

    LOGGER.debug("Proxy request id=%s payload=%s", request_id, json.dumps(payload))
    user_query = extract_user_query(payload, endpoint)
    context_block = ""

    # Use the fast classifier to decide whether to run recall.
    router: VoiceRouter | None = getattr(request.app.state, "voice_router", None)
    if user_query and router is not None:
        classify_started = time.perf_counter()
        try:
            classification = await router.quick_classify(user_query)
        except Exception as exc:
            LOGGER.warning("Classifier failed, falling back to SIMPLE: %s", exc)
            classification = "SIMPLE"
        classify_elapsed = (time.perf_counter() - classify_started) * 1000
        LOGGER.info(
            "Proxy classification id=%s result=%s query_chars=%d in %.1fms",
            request_id,
            classification,
            len(user_query),
            classify_elapsed,
        )

        if classification in ["COMPLEX_PERSONAL", "COMPLEX_GENERAL"]:
            context_started = time.perf_counter()
            try:
                # Bound recall time so requests aren't held up indefinitely.
                async with asyncio.timeout(router.config.context_timeout_seconds):
                    context_block = await memory_manager.build_context_block(user_query)
            except Exception as exc:
                LOGGER.warning("Context retrieval failed: %s", exc)
                context_block = ""
            context_elapsed = (time.perf_counter() - context_started) * 1000
            LOGGER.info(
                "Proxy context id=%s used=%s chars=%d in %.1fms",
                request_id,
                bool(context_block),
                len(context_block),
                context_elapsed,
            )
        else:
            LOGGER.info(
                "Proxy context id=%s skipped_for_classification=%s",
                request_id,
                classification,
            )
    elif user_query:
        # No router available (legacy) — attempt recall but bound by env timeout.
        context_started = time.perf_counter()
        try:
            async with asyncio.timeout(VOICE_CONTEXT_TIMEOUT_SECONDS):
                context_block = await memory_manager.build_context_block(user_query)
        except Exception as exc:
            LOGGER.warning("Context retrieval failed (no router): %s", exc)
            context_block = ""
        context_elapsed = (time.perf_counter() - context_started) * 1000
        LOGGER.info(
            "Proxy context id=%s legacy_router_path used=%s chars=%d in %.1fms",
            request_id,
            bool(context_block),
            len(context_block),
            context_elapsed,
        )
    else:
        LOGGER.info("Proxy request id=%s has_no_user_query", request_id)

    outgoing_payload = inject_context(payload, endpoint, context_block) if context_block else payload
    upstream_url = f"{OLLAMA_BASE_URL}/api/{endpoint}"
    LOGGER.info(
        "Proxy upstream start id=%s endpoint=%s stream=%s context_injected=%s",
        request_id,
        endpoint,
        stream,
        bool(context_block),
    )

    if stream:
        upstream_started = time.perf_counter()
        try:
            upstream_request = http_client.build_request("POST", upstream_url, json=outgoing_payload)
            upstream_response = await http_client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            LOGGER.exception("Streaming request to Ollama failed")
            return JSONResponse(
                status_code=502,
                content={"error": "Unable to reach Ollama", "details": str(exc)},
            )

        upstream_elapsed = (time.perf_counter() - upstream_started) * 1000
        LOGGER.info(
            "Proxy upstream accepted id=%s status=%s in %.1fms",
            request_id,
            upstream_response.status_code,
            upstream_elapsed,
        )

        if upstream_response.status_code >= 400:
            error_body = await upstream_response.aread()
            await upstream_response.aclose()
            return Response(
                content=error_body,
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get("content-type", "application/json"),
            )

        async def line_stream() -> AsyncIterator[bytes]:
            collected_chunks: list[str] = []
            chunk_count = 0
            first_chunk_ms: float | None = None
            try:
                async for line in upstream_response.aiter_lines():
                    if not line:
                        continue
                    chunk_count += 1
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - request_started) * 1000
                        LOGGER.info(
                            "Proxy first stream line id=%s at %.1fms",
                            request_id,
                            first_chunk_ms,
                        )
                    try:
                        decoded = json.loads(line)
                        chunk = extract_assistant_text(decoded, endpoint)
                        if chunk:
                            collected_chunks.append(chunk)
                    except json.JSONDecodeError:
                        LOGGER.debug("Skipping non-JSON streaming line")
                    yield f"{line}\n".encode("utf-8")
            finally:
                await upstream_response.aclose()
                total_elapsed = (time.perf_counter() - request_started) * 1000
                LOGGER.info(
                    "Proxy stream complete id=%s lines=%d chars=%d total=%.1fms",
                    request_id,
                    chunk_count,
                    sum(len(c) for c in collected_chunks),
                    total_elapsed,
                )
                if user_query and collected_chunks:
                    asyncio.create_task(
                        store_interaction_if_possible(
                            request,
                            user_query,
                            "".join(collected_chunks),
                        )
                    )

        return StreamingResponse(
            line_stream(),
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/x-ndjson"),
        )

    upstream_started = time.perf_counter()
    try:
        upstream_response = await http_client.post(upstream_url, json=outgoing_payload)
    except httpx.HTTPError as exc:
        LOGGER.exception("Non-streaming request to Ollama failed")
        return JSONResponse(
            status_code=502,
            content={"error": "Unable to reach Ollama", "details": str(exc)},
        )
    upstream_elapsed = (time.perf_counter() - upstream_started) * 1000
    LOGGER.info(
        "Proxy upstream done id=%s status=%s in %.1fms",
        request_id,
        upstream_response.status_code,
        upstream_elapsed,
    )

    if upstream_response.status_code >= 400:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
        )

    try:
        response_payload = upstream_response.json()
    except json.JSONDecodeError:
        response_payload = {}

    assistant_text = extract_assistant_text(response_payload, endpoint)
    if user_query and assistant_text:
        asyncio.create_task(store_interaction_if_possible(request, user_query, assistant_text))

    total_elapsed = (time.perf_counter() - request_started) * 1000
    LOGGER.info(
        "Proxy request complete id=%s endpoint=%s response_chars=%d total=%.1fms",
        request_id,
        endpoint,
        len(assistant_text),
        total_elapsed,
    )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get("content-type", "application/json"),
    )


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "ollama-hindsight-proxy",
        "status": "ok",
        "upstream": OLLAMA_BASE_URL,
        "memory": HINDSIGHT_HOST,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "ollama_base_url": OLLAMA_BASE_URL,
        "hindsight_host": HINDSIGHT_HOST,
    }


@app.get("/api/version")
async def api_version(request: Request) -> Response:
    return await passthrough_get(request, "/api/version")


@app.get("/api/tags")
async def api_tags(request: Request) -> Response:
    return await passthrough_get(request, "/api/tags")


@app.post("/api/show")
async def api_show(request: Request) -> Response:
    return await passthrough_post(request, "/api/show")


@app.post("/api/pull")
async def api_pull(request: Request) -> Response:
    return await passthrough_post_streaming(request, "/api/pull")


@app.post("/api/generate")
async def api_generate(request: Request) -> Response:
    return await proxy_generation_request(request, endpoint="generate")


@app.post("/api/chat")
async def api_chat(request: Request) -> Response:
    return await proxy_generation_request(request, endpoint="chat")


@app.post("/api/voice/handle")
async def api_voice_handle(request: Request) -> Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload"})

    transcription = _normalize_content(payload.get("transcription"))
    if not transcription:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'transcription' in request body"},
        )

    router: VoiceRouter = request.app.state.voice_router
    result = await router.handle_voice_input(transcription)
    return JSONResponse(status_code=200, content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PROXY_PORT, reload=False)
