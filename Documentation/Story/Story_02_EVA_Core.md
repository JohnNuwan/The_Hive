# Story 02: E.V.A. Core & API (The Brain)

## 📌 Context
The central nervous system. It orchestrates the Experts and handles communication with the outside world.

## 🎯 Objectives
- Setup FastAPI Gateway.
- Implement the LangGraph Router.
- Connect local LLM inference.

## 📋 Epic/Tasks Breakdown

### TASK-02-01: API Gateway (FastAPI)
- **Role**: Python Backend Dev
- **Description**: The entry point for all clients (Mobile, Web, MT5).
- **Endpoints**:
    - `POST /v1/chat/message`: Talk to EVA.
    - `POST /v1/system/status`: Health check.
    - `POST /v1/market/tick`: Ingest price data.
- **Acceptance Criteria**:
    - [ ] Swagger UI accessible.
    - [ ] Bearer Token Authentication (Simple static token for Phase 0).

### TASK-02-02: Orchestrator Skeleton (LangGraph)
- **Role**: AI Engineer
- **Description**: Define the graph state and nodes.
- **Nodes**:
    - `Router`: Decides which expert to call.
    - `Banker`: (Stub) Returns financial data.
    - `Chat`: (Stub) Returns Llama 3 response.
- **Acceptance Criteria**:
    - [ ] A query "What is the price of Gold?" routes to `Banker` node.
    - [ ] A query "Hello" routes to `Chat` node.

### TASK-02-03: Local Inference Engine (vLLM)
- **Role**: AI Engineer
- **Description**: Run the LLM server on the GPU.
- **Acceptance Criteria**:
    - [ ] vLLM Docker container running.
    - [ ] Llama-3 (Quantized) loaded.
    - [ ] Latency for "Hello World" < 1s.

### TASK-02-04: Memory Vector Store (Qdrant)
- **Role**: Data Engineer
- **Description**: Setup Qdrant DB.
- **Acceptance Criteria**:
    - [ ] Docker container running.
    - [ ] Python client can insert and retrieve a vector.
