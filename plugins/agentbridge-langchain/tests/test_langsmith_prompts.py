from __future__ import annotations

from agentbridge_langchain.langsmith_prompts import pull_prompt, push_prompt


class FakeClient:
    def __init__(self):
        self.pull_calls = []
        self.push_calls = []

    def pull_prompt(self, identifier, **kwargs):
        self.pull_calls.append((identifier, kwargs))
        return {"identifier": identifier}

    def push_prompt(self, identifier, **kwargs):
        self.push_calls.append((identifier, kwargs))
        return "commit-1"


def test_pull_prompt_preserves_identifier_and_native_options():
    client = FakeClient()

    result = pull_prompt("team/refunds:production", client=client)

    assert result == {"identifier": "team/refunds:production"}
    assert client.pull_calls == [
        (
            "team/refunds:production",
            {"include_model": True, "skip_cache": False},
        )
    ]


def test_push_prompt_creates_a_versioned_commit():
    client = FakeClient()

    commit = push_prompt(
        "team/refunds",
        prompt="native-prompt",
        client=client,
        tags=["production"],
        commit_description="Refund policy update",
    )

    assert commit == "commit-1"
    assert client.push_calls[0] == (
        "team/refunds",
        {
            "object": "native-prompt",
            "parent_commit_hash": "latest",
            "tags": ["production"],
            "description": None,
            "commit_description": "Refund policy update",
        },
    )
