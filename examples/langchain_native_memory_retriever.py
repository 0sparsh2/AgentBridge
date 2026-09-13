"""Use native LangChain/LangGraph memory and retriever objects with AgentBridge.

This example stays offline by using AgentBridge's plugin-only
``agentbridge/offline`` model string, but the extension config contains real
LangChain/LangGraph objects:

- a LangChain ``BaseRetriever`` subclass
- a LangGraph ``InMemorySaver`` checkpointer
- a LangGraph ``InMemoryStore``

Those native objects remain extension-level because memory and retrieval
semantics differ across frameworks.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agentbridge import AgentSpec, run_agent
from agentbridge.extensions.langchain import LangChainExtension


class RefundPolicyRetriever(BaseRetriever):
    """Tiny in-memory retriever for local examples and tests."""

    docs: list[Document]

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        del kwargs
        query = query.lower()
        return [doc for doc in self.docs if "refund" in query or "refund" in doc.page_content.lower()]


retriever = RefundPolicyRetriever(
    docs=[
        Document(
            page_content="Refunds over $100 require manager approval.",
            metadata={"source": "refund_policy"},
        )
    ]
)

agent = LangChainExtension.with_config(
    AgentSpec(
        name="langchain_support_agent",
        instructions="Answer support questions using refund policy context.",
        model="agentbridge/offline",
    ),
    memory="langgraph_in_memory_checkpointer",
    retrievers=[retriever],
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
    metadata={"example": "langchain_native_memory_retriever"},
)


if __name__ == "__main__":
    result = run_agent(
        agent,
        framework="langchain",
        input="Can I refund order A123?",
        session_id="example-thread",
    )
    print(result.output)
    print(result.metadata["extension_summary"])
