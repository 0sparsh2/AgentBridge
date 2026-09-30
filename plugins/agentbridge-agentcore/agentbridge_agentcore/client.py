"""Small lazy AWS client for AgentCore Runtime and service operations."""

from __future__ import annotations

import json
from typing import Any


class AgentCoreClient:
    """Call AgentCore Runtime or any installed AgentCore service operation.

    boto3 is loaded only when a request is made. ``call`` intentionally exposes
    the native operation surface so newly released AgentCore APIs do not require
    an AgentBridge release before they can be used.
    """

    def __init__(self, *, region: str | None = None, session: Any | None = None) -> None:
        self.region = region
        self._session = session

    def _session_or_default(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AgentCore requires boto3. Install `agentbridge-agentcore[aws]`."
            ) from exc
        return boto3.Session(region_name=self.region)

    def call(self, service: str, operation: str, **kwargs: Any) -> Any:
        """Invoke a native AgentCore client operation by service and method name."""

        client = self._session_or_default().client(service, region_name=self.region)
        method = getattr(client, operation, None)
        if not callable(method):
            raise AttributeError(f"AgentCore service {service!r} has no operation {operation!r}.")
        return method(**kwargs)

    def invoke_runtime(
        self,
        *,
        runtime_arn: str,
        session_id: str,
        payload: dict[str, Any],
        qualifier: str | None = None,
        trace_id: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "agentRuntimeArn": runtime_arn,
            "runtimeSessionId": session_id,
            "payload": json.dumps(payload).encode("utf-8"),
        }
        if qualifier:
            kwargs["qualifier"] = qualifier
        if trace_id:
            kwargs["traceId"] = trace_id
        response = self.call("bedrock-agentcore", "invoke_agent_runtime", **kwargs)
        body = response.get("response") if isinstance(response, dict) else response
        if hasattr(body, "read"):
            body = body.read()
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body
        return body
