# AgentBridge Visual System Maps

This page collects the high-level visuals for explaining AgentBridge in articles, README sections, architecture reviews, and future GitHub Pages posts.

## Market Position

```mermaid
flowchart LR
    app["Application code"]
    agentbridge["AgentBridge\napp-to-framework compatibility"]
    frameworks["Agent frameworks\nLangGraph, CrewAI, Pydantic AI,\nLangChain, Strands, ADK"]
    litellm["LiteLLM-style model strings\nprovider routing"]
    agui["AG-UI-shaped events\nfrontend compatibility"]

    app --> agentbridge
    agentbridge --> frameworks
    agentbridge --> agui
    frameworks --> litellm
```

AgentBridge sits above framework runtime selection and below application code. It does not replace AG-UI or LiteLLM; it complements them.

## Migration Wedge

```mermaid
flowchart TD
    prototype["Prototype agent\nrole, task, tool, prompt"]
    contract["AgentSpec contract\ninstructions, tools, output, metadata"]
    compare["Compare backends\nmock, pydantic_ai, langgraph, plugins"]
    report["Scenario report\noutputs, events, capabilities, gaps"]
    production["Production runtime\nstate, approval, tracing, deployment"]

    prototype --> contract
    contract --> compare
    compare --> report
    report --> production
```

The first wedge is migration. Teams should be able to capture a working agent shape, compare framework fit, and move toward a production runtime with less rewrite risk.

## Capability Layers

```mermaid
flowchart TB
    core["Portable core\nAgentSpec, ToolSpec, RunInput, RunResult, AgentEvent"]
    caps["Capability metadata\nfull, partial, extension, native_only, unsupported"]
    extensions["Framework extensions\nLangGraph, CrewAI, Pydantic AI, LangChain, Strands, ADK"]
    native["Native escape hatches\nraw framework objects and provider controls"]

    core --> caps
    caps --> extensions
    extensions --> native
```

The goal is not to flatten every framework into a weak common denominator. The goal is to make portability explicit while still preserving framework-specific power.

## Plugin Boundary

```mermaid
flowchart LR
    core["agentbridge core\nsmall install path"]
    optional["core optional extras\nclean local tests"]
    external["external plugins\nheavy or fast-moving frameworks"]
    cloud["credentialed smoke tests\nhosted services and deployments"]

    core --> optional
    core --> external
    external --> cloud
```

Core stays small. Frameworks with heavier dependencies, hosted services, or fast-moving constraints live as plugins until they are stable enough to justify a tighter integration.

## Status Graduation

```mermaid
stateDiagram-v2
    [*] --> Scaffold
    Scaffold --> Partial: native adapter plus contract tests
    Partial --> VerifiedLocally: local dependencies and no-key tests pass
    Partial --> CredentialVerified: hosted smoke tests pass
    VerifiedLocally --> Complete: deep feature matrix covered
    CredentialVerified --> Complete: deployment/eval/tracing paths verified
    Complete --> Maintenance: version policy and conformance kept current
```

Status labels should change only when evidence changes. A single provider smoke test proves one route, not every framework feature.
