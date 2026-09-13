"""Tiny MCP stdio server used by Strands MCP integration tests."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer


server = MCPServer(name="agentbridge-strands-test")


@server.tool(name="check_order")
def check_order(order_id: str) -> str:
    """Return a deterministic order lookup result."""

    return f"found:{order_id}"


if __name__ == "__main__":
    server.run(transport="stdio")
