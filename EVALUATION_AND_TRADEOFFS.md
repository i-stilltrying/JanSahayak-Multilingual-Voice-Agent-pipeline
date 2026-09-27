# Evaluation and Architectural Trade-offs

This document covers the evaluation methodology and architectural decisions for the JanSahayak voice agent, as requested for the submission.

## 1. How I Evaluated the Agent

To make sure this agent could actually handle real-world conversations without breaking, I avoided relying solely on manual testing and built a three-pronged evaluation approach:

*   **Automated Multi-Turn Scenario Harness:** I built a custom test runner (`evaluation/runner.py`) that executes predefined multi-turn conversations against the `ConversationManager`. By isolating the session state, I could programmatically assert whether the agent was selecting the right tools, passing the correct Pydantic arguments, completing the workflow, and strictly adhering to the requested language across an 8-turn conversation.
*   **Granular Latency Instrumentation:** Real-time voice is all about latency, so I instrumented every pipeline stage. I implemented a `TurnMetrics` tracker that logs exact timestamps using `time.perf_counter()` to measure STT processing, LLM reasoning, Tool execution, and TTS synthesis. This allowed me to track the true End-to-End Time-To-First-Audio (TTFA) and isolate network jitter from actual code bottlenecks.
*   **Targeted Unit & Integration Testing (12 Modules):** I wrote a 12-module pytest suite focusing heavily on the failure points of voice agents. I specifically tested the generation ID increment logic to ensure partial text is discarded during a barge-in, mocked the SQLite atomic updates to guarantee no double-booking of callback slots, and verified that the deterministic eligibility engine correctly coerces slots before returning a status.

### Key Learnings

*   **LLMs Should Extract, Not Decide:** I quickly learned that relying on an LLM to evaluate complex government eligibility rules leads to hallucinations. Offloading the actual decision-making to a deterministic Python rule engine while using the LLM strictly for entity extraction and conversational empathy is far more reliable. 
*   **Barge-in is a State Problem, Not Just Audio:** Handling interruptions isn't just about stopping playback. I learned that you have to aggressively guard the conversation history. If the user interrupts the agent mid-sentence, you must immediately increment a generation ID, cancel the TTS asyncio context manager, and discard the partial LLM text so it doesn't corrupt the canonical session memory.
*   **Two-Call Tool Routing is Safer:** I learned that trying to stream a response while simultaneously deciding if a tool needs to be called is incredibly brittle. Splitting the LLM interaction into two calls—one non-streaming call to confidently route to a tool, and a second streaming call to talk to the user—drastically improves tool accuracy.

```

```

## 2. The Biggest Trade-offs & Future Improvements

Building a real-time voice pipeline from scratch meant making several strict architectural trade-offs to balance reliability with developer velocity.

### The Two-Call LLM Pattern vs. Pure Streaming

The biggest trade-off I made was implementing a "two-call" LLM pattern. Call 1 is non-streaming with `tool_choice="auto"` to determine if a tool (like eligibility or knowledge search) is needed. Only after the tool executes do I initiate Call 2, which streams the audio back to the user.

* **The Trade-off:** This adds inherent latency to the pipeline because I wait for the first LLM call to fully complete before fetching tool data and starting the audio stream.
* **Why I did it:** It completely eliminates mid-stream tool hallucination and ensures every tool argument is fully validated via Pydantic before execution.

### In-Memory Keyword Retrieval vs. Vector Embeddings (RAG)

Instead of spinning up a vector database and an embedding model, I built an in-memory keyword scorer that loads the 5 scheme JSON files at startup.

* **The Trade-off:** It lacks semantic understanding (e.g., matching "money" to "financial aid").
* **Why I did it:** For a small, 5-scheme knowledge base, a weighted keyword scorer executes in less than 1 millisecond and requires zero GPU overhead, keeping the system exceptionally lightweight.

### Single-Process State vs. Distributed Memory

Session state (`ConversationState`) is held entirely in the memory of a single Uvicorn worker.

* **The Trade-off:** If the server restarts, all active conversations are lost. To scale this horizontally across multiple containers, it would require strict sticky sessions.
* **Why I did it:** It drastically simplifies the WebSocket orchestration and eliminates the need for an external Redis dependency.

### What I Would Do Differently With More Time

If I were to scale this into a production system serving thousands of citizens, I would focus on three immediate upgrades:

1. **Migrate Audio Transport to WebRTC (UDP):** WebSockets operate over TCP, which suffers from head-of-line blocking if network packets drop. Moving to WebRTC would allow audio packets to drop gracefully, preventing the stuttering and delayed VAD cuts often seen on slower Indian cellular networks.
2. **Move Session State to Redis:** I would refactor the `WebSocketHandler` to persist `ConversationState` into Redis. This would allow the backend to scale horizontally behind a load balancer without dropping user sessions if a specific worker crashes.
3. **Implement Streaming RAG:** I would replace the in-memory keyword retriever with a lightweight vector database (like Qdrant) and implement an asynchronous RAG pipeline that fetches context while the user is still speaking, further driving down the time-to-first-audio.
