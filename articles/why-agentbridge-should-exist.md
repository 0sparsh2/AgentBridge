# Why AgentBridge Should Exist

Agent frameworks are getting better quickly, but they are also getting more different from each other.

That is good for innovation. It is hard for application teams.

A team might start with CrewAI because roles and tasks make the prototype easy to explain. Later, the same team might need LangGraph because production workflows require durable state, checkpoints, interrupts, and replay. Another team might prefer Pydantic AI because typed Python outputs and validation are the center of the application. A platform team might standardize on OpenAI Agents SDK, Strands, Google ADK, or another framework because it fits their provider, deployment, or observability stack.

The problem is not that one framework is right and the others are wrong. The problem is that application code often becomes tightly coupled to whichever framework was chosen first.

AgentBridge exists to make that coupling less painful.

## The Pattern

Two useful standardization patterns already exist in the AI ecosystem:

- LiteLLM gives developers a model-provider compatibility layer.
- AG-UI gives agentic apps a frontend event protocol.

AgentBridge aims to sit in the missing middle:

> LiteLLM standardizes model access. AG-UI standardizes agent-to-frontend events. AgentBridge standardizes app-to-agent-framework compatibility.

The goal is not to flatten every framework into a weak lowest-common-denominator wrapper. That would erase exactly the reasons these frameworks exist.

The goal is to define a useful shared contract for the application layer:

- What is the agent?
- What tools can it use?
- What input did it receive?
- What output did it produce?
- What events happened during execution?
- Which backend capabilities are available?
- Which framework-specific settings were used?
- What raw backend object is preserved for advanced escape hatches?

## A Simple Example

The portable layer should stay boring:

```python
from agentbridge import AgentSpec, ToolSpec, run_agent


def check_order(order_id: str) -> str:
    """Return refund eligibility for an order."""
    return f"Order {order_id} is eligible for a refund."


agent = AgentSpec(
    name="refund_agent",
    instructions="Decide whether a customer is eligible for a refund.",
    model="openai/gpt-5",
    tools=[ToolSpec.from_function(check_order)],
)

result = run_agent(
    agent,
    framework="langgraph",
    input="Customer says order A123 was double charged.",
)

print(result.output)
```

That same `AgentSpec` can be tested against a mock backend, run through LangGraph, validated with Pydantic AI, or sent to an external plugin such as direct LangChain, OpenAI Agents, Strands, or Google ADK when the dependency is installed.

The application code should not need to rewrite every tool, output parser, event handler, and result object just to evaluate a different runtime.

## The Important Nuance

AgentBridge should not pretend that every backend supports every feature equally.

LangGraph is strong at graph orchestration and checkpoints. Pydantic AI is strong at typed outputs. CrewAI is strong at role/task prototyping. Strands is strong in AWS-oriented agent paths. Google ADK has strong session, memory, and deployment concepts. OpenAI Agents has a close mapping to handoffs, guardrails, approvals, and runner results. LangChain has a large ecosystem around agents, tools, middleware, retrievers, and LangSmith-style observability.

Those differences matter.

AgentBridge handles this through capability metadata and extension namespaces:

- Capability metadata says what a backend supports: `full`, `partial`, `extension`, `native_only`, or `unsupported`.
- Extension namespaces preserve framework-specific concepts without stuffing them into the portable core.
- Raw backend results remain available for advanced users.

That is the core design principle:

> Common things should be common. Framework-specific strengths should be visible, supported, and honest.

## Who Needs This?

AgentBridge is useful for teams that are asking questions like:

- Can we prototype in one framework and move to another later?
- Can we compare agent runtimes without rewriting the app?
- Can frontend event handling stay stable while backend agent frameworks change?
- Can platform teams enforce a shared agent contract across teams?
- Can adapter authors support a framework without bloating the core package?
- Can migration risk be reduced before committing to a production runtime?

If the answer to any of those is yes, a compatibility layer is valuable.

## What AgentBridge Is Not

AgentBridge is not a hosted gateway.

It is not a model-provider router. LiteLLM already points in that direction.

It is not an AG-UI replacement. AgentBridge can emit AG-UI-shaped events, but AG-UI owns the frontend protocol space.

It is not a claim that every framework can be reduced to the same object model.

AgentBridge is a Python SDK and plugin system for defining, running, comparing, and migrating agents across frameworks while keeping capability differences explicit.

## The Bet

The bet behind AgentBridge is that agent framework choice will keep changing.

Teams will not want to rewrite their whole application every time the best runtime changes. They will want a stable app-facing contract, honest capability reports, migration tools, and native escape hatches.

That is what AgentBridge is trying to build.

Start with the [README](../README.md), then read the [Vision And Target State](../docs/vision.md) and [Capability Coverage Strategy](../docs/capability_coverage.md).
