# ollama-proxy

## System Role
You are an expert Python developer specializing in FastAPI, asynchronous networking, and LLM orchestration.

## Project Goal
This repository provides a FastAPI middleware proxy between Home Assistant voice workflows and a local Ollama server, with Hindsight-backed retrieval and async interaction storage.

## Current Architecture (Implemented)
1. Proxy API compatibility:
    - Endpoints exposed for Ollama-compatible usage: `/api/generate`, `/api/chat`, `/api/show`, `/api/tags`, `/api/version`.
    - Requests are relayed to `OLLAMA_BASE_URL` using `httpx.AsyncClient`.
2. Hindsight memory integration:
    - Retrieval is handled by `MemoryManager.build_context_block()` from `memory_manager.py`.
    - Streaming and non-streaming responses store interactions asynchronously via `MemoryManager.store_interaction()`.
    - Hindsight client operations currently use `recall(...)` for retrieval and `retain(...)` for storage.
3. Voice routing pipeline (Fast Path vs Deep Path):
    - Implemented in `voice_router.py`.
        - Uses a small classifier model (default `llama3.2:1b`) to return `SIMPLE`, `COMPLEX_PERSONAL`, or `COMPLEX_GENERAL`.
    - Fallback heuristic on failure/timeout:
      - `len(text.split()) <= 3` => `SIMPLE`
            - first-person markers (`my`, `me`, `I`, etc.) => `COMPLEX_PERSONAL`
            - question words or `?` => `COMPLEX_GENERAL`
    - `SIMPLE` path: direct Ollama streaming to Piper sink without memory lookup.
        - `COMPLEX_GENERAL` path: direct Ollama streaming without memory lookup.
        - `COMPLEX_PERSONAL` path: buffer phrase playback and memory retrieval run concurrently, then response is streamed with optional context.

## Important Files
- `main.py`: app lifecycle, HTTP proxy behavior, memory injection, and `/api/voice/handle` endpoint.
- `memory_manager.py`: Hindsight wrapper and context block generation.
- `voice_router.py`: classification, routing, streaming, and buffer playback orchestration.

## Runtime Endpoints
- `GET /health`: service health.
- `POST /api/generate`: Ollama generate proxy with optional memory injection.
- `POST /api/chat`: Ollama chat proxy with optional memory injection.
- `POST /api/voice/handle`: voice routing entry point expecting `{ "transcription": "..." }`.

## Environment Variables
Core:
- `OLLAMA_BASE_URL`
- `HINDSIGHT_HOST`
- `PROXY_PORT`
- `HINDSIGHT_MAX_MEMORIES`
- `HINDSIGHT_BANK`
- `LOG_LEVEL`

Voice routing:
- `VOICE_CLASSIFIER_MODEL` (default `llama3.2:1b`)
- `VOICE_RESPONSE_MODEL` (default `llama3`)
- `VOICE_CLASSIFIER_TIMEOUT_SECONDS` (default `1.2`)
- `VOICE_CONTEXT_TIMEOUT_SECONDS` (default `3.5`)
- `VOICE_BUFFER_PHRASE` (default `Let me check...`)

## Piper Integration Notes
`main.py` currently wires placeholder async functions:
- `piper_stream_chunk(chunk: str)`
- `piper_play_phrase(text: str)`
- `piper_stop_playback()`

These are stubs that log activity and should be replaced with real Piper playback/streaming integration.

## Behavioral Constraints
- Prefer fail-open behavior when memory retrieval fails.
- Preserve low-latency streaming behavior.
- Do not block response paths on memory storage.
- Avoid introducing blocking calls in async paths.

## Guidance for Future Agents
- Keep compatibility with Home Assistant/Ollama payload formats.
- Maintain both streaming and non-streaming support.
- If changing retrieval semantics, preserve graceful fallback to raw prompts.
- If changing voice routing, keep classifier timeout + heuristic fallback to avoid latency spikes.
- Add tests for routing decisions and stream forwarding whenever behavior changes.
- Preserve explicit cleanup of `httpx` and `hindsight_client` clients to avoid unclosed `aiohttp` session/connector warnings.

## How to Test Quickly
1. Start the service:

```bash
source .venv/bin/activate
python main.py
```

2. Run tests:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

3. Health check:

```bash
curl -s http://127.0.0.1:8000/health | jq .
```

4. Non-streaming chat proxy check:

```bash
curl -s http://127.0.0.1:8000/api/chat \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "llama3",
        "stream": false,
        "messages": [
            {"role": "user", "content": "Say hello in one short sentence."}
        ]
    }' | jq .
```

5. Streaming chat proxy check:

```bash
curl -N http://127.0.0.1:8000/api/chat \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "llama3",
        "stream": true,
        "messages": [
            {"role": "user", "content": "Count from 1 to 5."}
        ]
    }'
```

6. Voice routing check (SIMPLE path candidate):

```bash
curl -s http://127.0.0.1:8000/api/voice/handle \
    -H 'Content-Type: application/json' \
    -d '{"transcription":"Hey there"}' | jq .
```

7. Voice routing check (COMPLEX_PERSONAL path candidate):

```bash
curl -s http://127.0.0.1:8000/api/voice/handle \
    -H 'Content-Type: application/json' \
    -d '{"transcription":"What did I ask about my calendar yesterday?"}' | jq .
```

8. Verify logs for latency and routing decisions:
    - Look for classifier output (`SIMPLE`, `COMPLEX_PERSONAL`, or `COMPLEX_GENERAL`).
     - Look for path logs (`FAST` or `DEEP`).
     - Confirm timing fields are reported in endpoint responses.

Notes:
- If `jq` is not installed, remove `| jq .` from commands.
- Piper hooks in `main.py` are placeholders; endpoint testing still validates routing and streaming logic.