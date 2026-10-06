import copy
import json

import pytest

from scripts import check_langfuse_api as checker


def schema_fixture():
    baseline = json.loads(checker.BASELINE.read_text())
    adopted = baseline["contracts"]
    paths = {}
    for path, contract in adopted.items():
        parameters = [{"name": name, "in": "query", "required": value["required"],
                       "schema": {key: item for key, item in value.items() if key != "required"}}
                      for name, value in contract["query_pagination"].items()]
        parameters += [{"name": name, "in": "query", "required": True, "schema": {"type": "string"}}
                       for name in contract["required_query"] if name not in contract["query_pagination"]]
        meta = {"type": "object", "required": contract["metadata_required"],
                "properties": copy.deepcopy(contract["metadata"])}
        response = {"type": "object", "required": contract["response_required"],
                    "properties": {"data": {"type": contract["data_type"]}, "meta": meta}}
        paths[path] = {"get": {"parameters": parameters, "responses": {
            "200": {"content": {"application/json": {"schema": response}}},
        }}}
    for key, contract in baseline["operations"].items():
        method, path = key.split(" ", 1)
        operation = paths.setdefault(path, {}).setdefault(method.lower(), {})
        operation["parameters"] = [
            {"name": name, "in": value["in"], "required": value["required"],
             "schema": {part: item for part, item in value.items() if part not in ("in", "required")}}
            for name, value in contract["path_and_query"].items()
        ]
        if contract["request"] is not None:
            request = contract["request"]
            operation["requestBody"] = {"required": request["required"], "content": {
                "application/json": {"schema": {"type": "object", "required": request["required"],
                                              "properties": copy.deepcopy(request["properties"])}}}}
        operation["responses"] = {
            status: ({"content": {"application/json": {"schema": {
                "type": "object", "required": response["required"],
                "properties": copy.deepcopy(response["properties"]),
            }}}} if response is not None else {})
            for status, response in contract["successes"].items()
        }
    return {"paths": paths}, baseline


def test_adopted_pagination_contracts_distinguish_numbered_and_cursor_endpoints():
    spec, adopted = schema_fixture()
    assert checker.extract_contracts(spec) == adopted["contracts"]
    for path in checker.ENDPOINTS:
        keys = adopted["contracts"][path]["query_pagination"]
        if path.endswith(("prompts", "datasets", "dataset-items", "score-configs", "annotation-queues", "/items")):
            assert "page" in keys and "cursor" not in keys
            assert adopted["contracts"][path]["metadata_required"] == ["limit", "page", "totalItems", "totalPages"]
        else:
            assert "cursor" in keys and "page" not in keys
    report, drifted = checker.render_report(adopted["contracts"], checker.extract_contracts(spec))
    assert not drifted
    assert report.count("unchanged") == len(checker.ENDPOINTS)


def test_contract_extraction_resolves_native_refs_and_inherited_metadata():
    spec, adopted = schema_fixture()
    response = spec["paths"][checker.ENDPOINTS[0]]["get"]["responses"]["200"]["content"]["application/json"]
    meta = response["schema"]["properties"]["meta"]
    spec["components"] = {"schemas": {"Meta": meta, "Response": response["schema"]}}
    response["schema"] = {"$ref": "#/components/schemas/Response"}
    spec["components"]["schemas"]["Response"]["properties"]["meta"] = {
        "allOf": [{"$ref": "#/components/schemas/Meta"}],
    }
    assert checker.extract_contracts(spec) == adopted["contracts"]


@pytest.mark.parametrize("change", ["required_query", "cursor_type", "metadata", "response_required"])
def test_pagination_drift_reports_actual_field_diff(change):
    spec, adopted = schema_fixture()
    operation = spec["paths"][checker.ENDPOINTS[0]]["get"]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    if change == "required_query":
        operation["parameters"].append({"name": "newRequired", "in": "query", "required": True,
                                        "schema": {"type": "string"}})
    elif change == "cursor_type":
        next(item for item in operation["parameters"] if item["name"] == "cursor")["schema"]["type"] = "integer"
    elif change == "metadata":
        schema["properties"]["meta"]["properties"]["nextCursor"] = {"type": "string"}
    else:
        schema["required"] = ["data"]
    report, drifted = checker.render_report(adopted["contracts"], checker.extract_contracts(spec))
    assert drifted
    assert report.count("review needed") == 1
    assert "```diff" in report


def test_unrelated_description_drift_does_not_change_pagination_contract():
    spec, adopted = schema_fixture()
    original = copy.deepcopy(spec)
    spec["paths"][checker.ENDPOINTS[0]]["get"]["description"] = "New documentation"
    assert checker.extract_contracts(spec) == checker.extract_contracts(original) == adopted["contracts"]


def test_adopted_governance_write_contracts_match_native_operations():
    spec, adopted = schema_fixture()
    assert checker.extract_governance_operations(spec) == adopted["operations"]
    report, drifted = checker.render_operations_report(
        adopted["operations"], checker.extract_governance_operations(spec),
    )
    assert not drifted
    assert report.count("unchanged") == len(checker.GOVERNANCE_OPERATIONS)


@pytest.mark.parametrize("change", ["request_required", "path_type", "response_required"])
def test_governance_write_drift_reports_field_changes(change):
    spec, adopted = schema_fixture()
    operation = spec["paths"]["/api/public/annotation-queues/{queueId}/items"]["post"]
    if change == "request_required":
        operation["requestBody"]["content"]["application/json"]["schema"]["required"] = ["objectId"]
    elif change == "path_type":
        operation["parameters"][0]["schema"]["type"] = "integer"
    else:
        operation["responses"]["200"]["content"]["application/json"]["schema"]["required"] = ["id"]
    report, drifted = checker.render_operations_report(
        adopted["operations"], checker.extract_governance_operations(spec),
    )
    assert drifted
    assert report.count("review needed") == 1
    assert "```diff" in report


@pytest.mark.parametrize("reference", ["https://example.invalid/schema.json", "#/components/schemas/Cycle"])
def test_resolver_rejects_external_or_cyclic_references(reference):
    spec = {"components": {"schemas": {"Cycle": {"$ref": "#/components/schemas/Cycle"}}}}
    with pytest.raises(ValueError, match="reference"):
        checker.resolve_schema(spec, {"$ref": reference})


def test_cli_local_replay_and_drift_exit_status(tmp_path, monkeypatch):
    spec, _ = schema_fixture()
    source = tmp_path / "openapi.json"
    output = tmp_path / "report.md"
    source.write_text(json.dumps(spec))
    monkeypatch.setattr("sys.argv", ["check", "--schema", str(source), "--output", str(output), "--fail-on-drift"])
    assert checker.main() == 0
    del spec["paths"][checker.ENDPOINTS[0]]["get"]["parameters"][0]["schema"]["nullable"]
    source.write_text(json.dumps(spec))
    assert checker.main() == 1
    assert "review needed" in output.read_text()


def test_cli_fetch_failure_is_unverified_and_never_success(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("upstream unavailable")

    output = tmp_path / "report.md"
    monkeypatch.setattr(checker, "urlopen", fail)
    monkeypatch.setattr("sys.argv", ["check", "--output", str(output)])
    assert checker.main() == 2
    assert "unverified" in output.read_text()


def test_cli_missing_endpoint_is_a_failed_check(tmp_path, monkeypatch):
    source = tmp_path / "openapi.json"
    source.write_text('{"paths": {}}')
    monkeypatch.setattr("sys.argv", ["check", "--schema", str(source)])
    assert checker.main() == 2
