# Enterprise AI patterns

Production-style examples: streaming APIs, extraction, stateful agents, graph RAG, and focused “challenge” modules (hybrid search, routing, guardrails, HITL, tool use).

Shared Python dependencies are listed in [`requirements.txt`](./requirements.txt). Install per submodule as needed.

## Core demos

| Folder | Topic |
|--------|--------|
| [`01_fastapi_streaming`](./01_fastapi_streaming/) | Streaming LLM responses over FastAPI |
| [`02_structured_extraction`](./02_structured_extraction/) | Structured output / extraction patterns |
| [`03_langgraph_agent`](./03_langgraph_agent/) | LangGraph-style agent workflow (“newsroom” multi-agent pattern) |
| [`04_graph_rag_neo4j`](./04_graph_rag_neo4j/) | Graph RAG with Neo4j |

## Challenges (single-purpose reference implementations)

| Folder | Topic |
|--------|--------|
| [`challenge_01_hierarchical_summary`](./challenge_01_hierarchical_summary/) | Hierarchical summarization |
| [`challenge_02_hybrid_search`](./challenge_02_hybrid_search/) | BM25 + dense vector hybrid retrieval |
| [`challenge_04_cost_router`](./challenge_04_cost_router/) | Semantic cost / model routing |
| [`challenge_07_agentic_rag`](./challenge_07_agentic_rag/) | Agentic RAG loop |
| [`challenge_08_guardrails`](./challenge_08_guardrails/) | Safety and I/O filtering |
| [`challenge_10_human_in_the_loop`](./challenge_10_human_in_the_loop/) | Approval / HITL flows |
| [`challenge_11_code_interpreter`](./challenge_11_code_interpreter/) | Code interpreter tool use |

Each challenge folder has its own `README.md` with run notes where applicable.
