# ollama-proxy

**System Role:** You are an expert Python developer specializing in FastAPI, asynchronous networking, and LLM orchestration.

**Project Goal:** Create a Python-based middleware proxy that sits between Home Assistant’s `voice_assistant` component and a local Ollama instance. The proxy will integrate **Hindsight** as a memory layer to provide persistent, long-term context for home automation queries.

**Architecture Requirements:**
1.  **Proxy Layer:** Build a FastAPI server that mimics the Ollama API (specifically the `/api/generate` or `/api/chat` endpoints).
2.  **Hindsight Integration:**
    * **Retrieval:** For every incoming request, use Hindsight’s `search()` function to retrieve relevant memories using its hybrid strategies (semantic, BM25, graph, and temporal).
    * **Injection:** Augment the user's prompt by prepending the retrieved context before forwarding it to the actual Ollama endpoint.
    * **Storage:** Once Ollama responds, use Hindsight’s `add()` function to store the interaction (User Query + Assistant Response) asynchronously.
3.  **Environment Configuration:** Use a `.env` file to manage the `OLLAMA_BASE_URL`, `HINDSIGHT_HOST`, and `PROXY_PORT`.

**Logic Flow to Implement:**
* **Intercept:** Capture the JSON payload from Home Assistant.
* **Contextualize:** * Extract the current query.
    * Query Hindsight for memories.
    * Construct a "Context Block" (e.g., `Relevant past interactions: [Memories]. Current Request: [Query]`).
* **Relay:** Forward the modified payload to the real Ollama service.
* **Persist:** Log the exchange to Hindsight without blocking the response back to Home Assistant (ensure low latency).

**Technical Constraints:**
* **Latency:** Prioritize standard retrieval over "reflect" operations to keep voice response times snappy.
* **Error Handling:** If Hindsight is unavailable, the proxy should "fail open" and forward the raw query to Ollama so the smart home remains functional.
* **Streaming:** Ensure the proxy supports streaming responses if Home Assistant is configured for it.

**Deliverables:**
1.  A clean, modular `main.py` using FastAPI and `httpx`.
2.  A `memory_manager.py` wrapper for Hindsight logic.
3.  A `docker-compose.yml` to run the proxy alongside Hindsight and Ollama.

---

### A Quick Tip for Success
Since you mentioned Home Assistant's `voice_assistant` component, you will likely need to point your HA configuration to your new Proxy IP instead of the Ollama IP:

```yaml
# In Home Assistant configuration.yaml (Example)
llm_control:
  - platform: ollama
    url: "http://<YOUR_PROXY_IP>:8000" # Point here instead of 11434
```

### Why this works
By framing the prompt this way, you are giving the coding agent a **design pattern** (Middleware/Proxy) rather than just a list of features. This ensures the agent focuses on the networking aspect (handling requests/responses) and the data flow between the two databases (Hindsight for memory, Ollama for reasoning). 

Do you want to focus more on the **Temporal Reasoning** aspect in the first iteration, or just get the basic **Store/Retrieve** loop running?