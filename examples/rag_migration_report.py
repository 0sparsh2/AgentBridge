"""Generate a no-key LangChain-to-LangGraph RAG migration report."""

from __future__ import annotations

import json

from agentbridge import AgentSpec, get_adapter, run_agent
from agentbridge.extensions.langchain import LangChainExtension
from agentbridge.extensions.langgraph import LangGraphExtension
from examples.model_routes import MODEL_ROUTES


rag_agent = AgentSpec(
    name="support_rag_agent",
    instructions="Answer support questions using retrieved refund policy context.",
    model="agentbridge/offline",
)


def build_report() -> dict[str, object]:
    langchain_source = LangChainExtension.with_config(
        rag_agent,
        retrievers=["refund_policy_docs"],
        memory="conversation_buffer",
        callbacks=["langsmith"],
        checkpointer="native_checkpointer",
        store="native_vector_store",
        metadata={"migration_source": "langchain_rag"},
    )
    langgraph_target = LangGraphExtension.with_config(
        rag_agent,
        graph_name="support_rag_graph",
        node_name="answer_with_policy",
        include_context_in_output=True,
        enable_checkpointing=True,
        route_on_context_key="intent",
        routes={"refund": "answer_with_policy", "billing": "billing_handoff"},
    )
    return {
        "scenario": "support RAG migration",
        "source_framework": "langchain",
        "target_framework": "langgraph",
        "agent_spec": rag_agent.model_dump(mode="json"),
        "framework_extensions": {
            "langchain_source": langchain_source.backend_config["langchain"],
            "langgraph_target": langgraph_target.backend_config["langgraph"],
        },
        "model_routes": {
            "offline_ci": MODEL_ROUTES["offline_ci"]["model"],
            "local_ollama": MODEL_ROUTES["local_ollama"]["model"],
            "openrouter": MODEL_ROUTES["openrouter"]["model"],
            "custom_openai_compatible_gateway": MODEL_ROUTES["custom_openai_compatible_gateway"]["model"],
        },
        "offline_run_comparison": _offline_run_comparison(
            rag_agent,
            ["mock", "langchain", "langgraph"],
        ),
        "migration_notes": [
            "LangChain retriever, store, memory, callbacks, and checkpointer remain extension-specific.",
            "LangGraph target captures routing and checkpoint intent for production orchestration.",
            "Portable app code should rely on normalized RunResult output/events, not native RAG internals.",
        ],
    }


def _offline_run_comparison(agent: AgentSpec, backends: list[str]) -> dict[str, object]:
    comparison: dict[str, object] = {}
    for backend in backends:
        try:
            get_adapter(backend)
            result = run_agent(
                agent,
                backend=backend,
                input="What is the refund policy for double charges?",
                session_id="rag-migration-report",
                context={"intent": "refund", "retrieved_policy": "Double charges are refundable."},
            )
        except Exception as exc:  # pragma: no cover - depends on optional local plugins.
            comparison[backend] = {
                "available": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }
            continue
        comparison[backend] = {
            "available": True,
            "backend": result.backend,
            "output": _json_safe(result.output),
            "events": [event.type for event in result.events],
            "metadata_keys": sorted(result.metadata),
        }
    return comparison


def _json_safe(value: object) -> object:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return type(value).__name__


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
