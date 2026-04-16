Build a low-latency voice assistant routing pipeline using asyncio.

Stack:
- Whisper (already provides transcription as input string)
- Ollama (LLM inference via HTTP API or python client)
- Hindsight (memory/context retrieval module, assume async function exists)
- Piper (TTS playback, assume async streaming function exists)

Goal:
Implement a "Fast Path vs Deep Path" routing system to reduce perceived latency.

Core Requirements:

1. Router (Fast Classifier)
- Implement async function: quick_classify(text: str) -> str
- Use ONE of:
  a) Ollama small model (e.g., llama3.2:1b or llama3.2:1b)
  b) fallback heuristic if LLM fails
- Prompt must force SINGLE WORD output:
    "SIMPLE" or "COMPLEX"
- SIMPLE = greetings, short phrases, chit-chat
- COMPLEX = questions requiring facts, memory, or retrieval

2. Main Orchestrator
Implement:
    async def handle_voice_input(transcription: str)

Behavior:
- Call quick_classify()
- If "SIMPLE":
    -> immediately stream response from Ollama to Piper
    -> no memory lookup
- If "COMPLEX":
    -> concurrently:
        a) play a short buffer phrase via Piper (non-blocking)
        b) fetch context from hindsight.get_memory()
    -> after context is ready:
        -> call Ollama with transcription + context
        -> stream result to Piper

3. Streaming
- Implement stream_ollama_to_piper(prompt: str, context: str | None)
- Must:
    - stream tokens/chunks from Ollama
    - forward chunks incrementally to Piper
- Avoid waiting for full completion before TTS

4. Buffer Phrase Handling
- Implement:
    async def play_buffer_phrase(text: str)
- Should NOT block main pipeline
- Should stop or yield naturally when final response starts

5. Concurrency Design
- Use asyncio.create_task for parallel execution
- Ensure:
    - buffer playback does not delay final answer
    - context retrieval overlaps with buffer playback

6. Ollama Integration
- Use two models:
    - small model for classification
    - larger model for responses
- Example:
    ollama.generate(model="llama3.2:1b", prompt=...)
    ollama.chat(model="llama3", messages=[...], stream=True)

7. Heuristic Fallback (important)
If classifier fails or times out:
- If len(text.split()) <= 3 → return "SIMPLE"
- If text contains question words ("what", "when", "how", etc.) → "COMPLEX"

8. Code Structure
- Keep functions small and testable
- Use type hints
- Add logging for:
    - classification decision
    - path taken (FAST vs DEEP)
    - timing (latency measurement)

9. Example Flow

Input: "Hey there"
→ SIMPLE
→ direct LLM → Piper

Input: "What did I say about my daughter's rehab?"
→ COMPLEX
→ play "Let me check..."
→ fetch memory
→ LLM with context → Piper

10. Constraints
- Optimize for perceived latency, not just raw speed
- Avoid blocking calls
- Assume all external services are local (low network latency)

Return clean, runnable Python code.
