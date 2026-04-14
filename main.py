import asyncio
import copy
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from memory_manager import MemoryManager

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
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    app.state.memory_manager = MemoryManager(
        host=HINDSIGHT_HOST,
        bank=HINDSIGHT_BANK,
        max_memories=HINDSIGHT_MAX_MEMORIES,
    )
    LOGGER.info("Proxy starting with Ollama at %s", OLLAMA_BASE_URL)
    try:
        yield
    finally:
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


async def proxy_generation_request(request: Request, endpoint: str) -> Response:
    http_client: httpx.AsyncClient = request.app.state.http
    memory_manager: MemoryManager = request.app.state.memory_manager

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload"})

    user_query = extract_user_query(payload, endpoint)
    context_block = await memory_manager.build_context_block(user_query) if user_query else ""
    outgoing_payload = inject_context(payload, endpoint, context_block) if context_block else payload
    upstream_url = f"{OLLAMA_BASE_URL}/api/{endpoint}"

    if bool(payload.get("stream", False)):
        try:
            upstream_request = http_client.build_request("POST", upstream_url, json=outgoing_payload)
            upstream_response = await http_client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            LOGGER.exception("Streaming request to Ollama failed")
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

        async def line_stream() -> AsyncIterator[bytes]:
            collected_chunks: list[str] = []
            try:
                async for line in upstream_response.aiter_lines():
                    if not line:
                        continue
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

    try:
        upstream_response = await http_client.post(upstream_url, json=outgoing_payload)
    except httpx.HTTPError as exc:
        LOGGER.exception("Non-streaming request to Ollama failed")
        return JSONResponse(
            status_code=502,
            content={"error": "Unable to reach Ollama", "details": str(exc)},
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


@app.post("/api/generate")
async def api_generate(request: Request) -> Response:
    return await proxy_generation_request(request, endpoint="generate")


@app.post("/api/chat")
async def api_chat(request: Request) -> Response:
    return await proxy_generation_request(request, endpoint="chat")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PROXY_PORT, reload=False)
