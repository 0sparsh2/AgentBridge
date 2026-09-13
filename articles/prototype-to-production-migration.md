# From Prototype Agents To Production Runtimes

Most agent projects do not start with the final runtime.

They start with a prototype.

Someone builds a refund assistant, research helper, sales workflow, support triage agent, or internal tool. The first framework is usually chosen because it makes that prototype fast.

That is reasonable.

The problem comes later.

## The Prototype-To-Production Gap

A prototype agent often needs:

- Simple roles and tasks.
- A few Python tools.
- Fast iteration.
- Easy demos.
- Minimal infrastructure.

A production agent often needs:

- Durable state.
- Explicit workflow control.
- Human approval and resume.
- Typed outputs.
- Observability.
- Evals.
- Session and memory services.
- Deployment to a specific runtime.
- Frontend event stability.

Those are not always strengths of the same framework.

That means teams face a painful choice: keep stretching the prototype framework past its natural shape, or rewrite the application around a more production-oriented runtime.

AgentBridge is meant to make that transition less expensive.

## Migration As The First Wedge

The first practical use case for AgentBridge is not “write one agent and magically run it everywhere perfectly.”

The first wedge is migration:

> Define the application-facing agent contract once, then compare or move across backends while preserving tools, inputs, outputs, events, tests, and capability visibility.

For example, a team could:

1. Start with `mock` for no-key local testing.
2. Validate typed behavior with `pydantic_ai`.
3. Move stateful orchestration to `langgraph`.
4. Test a direct `langchain` adapter for an existing LangChain app.
5. Evaluate OpenAI Agents, Strands, or Google ADK through external plugins.
6. Use capability reports to decide what is portable, extension-level, or native-only.

The application code should change less than the runtime configuration.

## What Stays Stable

The shared app contract can keep these stable:

- Agent name and instructions.
- Tool definitions.
- Run input shape.
- Result shape.
- Event shape.
- Structured output schema.
- Backend capability inspection.
- Basic CLI validation and conformance checks.

This gives teams a common surface for tests and integration code.

## What Does Not Become Fake-Portable

Some things should not be forced into the common core too early.

LangGraph graph topology, CrewAI role/task semantics, LangChain retriever objects, Strands AgentCore deployment paths, OpenAI Agents approval stores, and Google ADK session services all have native shape.

AgentBridge handles these through extension namespaces and raw escape hatches:

```python
from agentbridge.extensions.langgraph import LangGraphExtension

agent = LangGraphExtension.with_config(
    agent,
    enable_checkpointing=True,
    route_on_context_key="intent",
    routes={"refund": "refund_node", "billing": "billing_node"},
)
```

This is not hiding the framework. It is documenting and containing the framework-specific part.

## A Real-World Example

Imagine a customer support agent.

The prototype needs:

- `check_order(order_id)`
- `issue_refund(order_id)`
- a prompt that decides refund eligibility
- a result that frontend code can display

Production later needs:

- approval before issuing large refunds
- durable session state
- audit logs
- routing between refund, billing, and escalation paths
- structured output for downstream systems
- streaming events for the UI

AgentBridge lets the team keep the portable agent definition and then evaluate which backend should own the production concerns.

LangGraph might own routing and checkpointing. OpenAI Agents might expose approval interruption diagnostics. Google ADK might model sessions and services. Strands might capture AWS deployment metadata. LangChain might be needed for existing retrievers and callbacks.

The point is not that every backend does every job.

The point is that the team can compare honestly before rewriting everything.

## What Success Looks Like

AgentBridge succeeds when:

- Developers can define one useful `AgentSpec`.
- The same spec can run across multiple backends.
- Capability reports explain what will and will not work.
- Migration helpers inspect existing projects conservatively.
- External plugins can support heavy frameworks without bloating the core.
- Conformance tests back every `full` capability claim.

This is why the project is deliberately building docs, tests, adapters, version policy, and issue tracking together.

The compatibility layer is not just code. It is also a set of promises about what works, what is partial, and what remains native-only.

See [examples/refund_agent.py](../examples/refund_agent.py), [examples/framework_extensions.py](../examples/framework_extensions.py), and the [Conformance docs](../docs/conformance.md) to explore the migration workflow.
