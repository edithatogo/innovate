"""Scenario Arrow exports require explicit outcomes and preserve consumer data."""

import json

import polars as pl
import pytest

import innovate


def envelope(field, rows):
    return {"operation": "execute_scenario_grid", "error": None, "result": {field: rows}}


EXPORTS = [
    pytest.param(innovate.scenario_executions, "executions", {"id": "a", "model_key": "bass"}, id="execution"),
    pytest.param(
        innovate.scenario_comparisons,
        "comparisons",
        {"baseline": "a", "alternative": "b"},
        id="pair",
    ),
    pytest.param(
        innovate.scenario_comparisons,
        "comparisons",
        {"pairs": [{"baseline": "a", "alternative": "b"}]},
        id="ensemble",
    ),
]


@pytest.mark.parametrize("export,field,identity", EXPORTS)
def test_scenario_export_requires_explicit_envelope_error(export, field, identity):
    response = envelope(field, [{**identity, "result": {"values": [1.0]}, "error": None}])
    del response["error"]
    with pytest.raises(ValueError):
        export(response)


@pytest.mark.parametrize("export,field,identity", EXPORTS)
@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param({}, id="missing-both"),
        pytest.param({"result": {"values": [1.0]}}, id="missing-error"),
        pytest.param({"error": None}, id="missing-result"),
        pytest.param({"result": None, "error": None}, id="empty-outcome"),
        pytest.param(
            {"result": {"values": [1.0]}, "error": {"code": "invalid_request"}},
            id="contradictory-outcome",
        ),
        pytest.param({"result": [1.0], "error": None}, id="array-result"),
        pytest.param({"result": False, "error": None}, id="scalar-result"),
        pytest.param({"result": None, "error": "failed"}, id="string-error"),
        pytest.param({"result": None, "error": []}, id="array-error"),
        pytest.param({"result": None, "error": False}, id="boolean-error"),
    ],
)
def test_scenario_export_rejects_incomplete_or_contradictory_outcomes(export, field, identity, outcome):
    with pytest.raises(ValueError):
        export(envelope(field, [{**identity, **outcome}]))

    # Rejection must not poison subsequent use of the installed native library.
    valid = {**identity, "result": {"values": [1.0]}, "error": None}
    recovered = export(envelope(field, [valid]))
    assert recovered["successful"].to_list() == [True]


def execution_rows():
    return [
        {
            "id": "z-β",
            "model_key": "bass",
            "result": {"values": [0.0, 2.5], "metadata": {"seed": 2**64 - 1, "note": None}},
            "error": None,
            "artifact": {"notes": "synthetic β", "optional": None},
        },
        {
            "id": "a",
            "model_key": "unknown",
            "result": None,
            "error": {"code": "unsupported_native_operation", "message": "synthetic failure"},
        },
        {
            "id": "m",
            "model_key": "bass",
            "result": {"status": "paused", "checkpoint": "opaque-β-18446744073709551615"},
            "error": None,
        },
    ]


def comparison_rows():
    return [
        {
            "baseline": "z-β",
            "alternative": "a",
            "result": {"difference": [-2.5, 1.5], "notes": {"draw_id": 2**64 - 1, "optional": None}},
            "error": None,
        },
        {
            "pairs": [{"baseline": "z-β", "alternative": "a"}],
            "result": None,
            "error": {"code": "invalid_comparison", "message": "synthetic failure"},
        },
    ]


def test_valid_scenario_exports_preserve_outcomes_order_and_nested_json():
    rows = execution_rows()
    frame = innovate.scenario_executions(envelope("executions", rows))
    assert frame.schema == {
        "scenario_id": pl.String,
        "model_key": pl.String,
        "successful": pl.Boolean,
        "execution_json": pl.String,
    }
    assert frame["scenario_id"].to_list() == ["z-β", "a", "m"]
    assert frame["successful"].to_list() == [True, False, False]
    assert [json.loads(row) for row in frame["execution_json"]] == rows

    rows = comparison_rows()
    frame = innovate.scenario_comparisons(envelope("comparisons", rows))
    assert frame.schema == {
        "comparison_index": pl.UInt64,
        "kind": pl.String,
        "successful": pl.Boolean,
        "comparison_json": pl.String,
    }
    assert frame["comparison_index"].to_list() == [0, 1]
    assert frame["kind"].to_list() == ["pair", "ensemble"]
    assert frame["successful"].to_list() == [True, False]
    assert [json.loads(row) for row in frame["comparison_json"]] == rows


@pytest.mark.parametrize("field", ["executions", "comparisons"])
@pytest.mark.parametrize("ipc_format", ["file", "stream"])
def test_optional_pyarrow_reads_and_roundtrips_native_scenario_exports(field, ipc_format):
    pa = pytest.importorskip("pyarrow", reason="requires the locked optional Arrow extra")
    rows = execution_rows() if field == "executions" else comparison_rows()
    export = innovate.scenario_executions if field == "executions" else innovate.scenario_comparisons
    json_column = "execution_json" if field == "executions" else "comparison_json"
    frame = export(envelope(field, rows))

    to_native_ipc = innovate.export_ipc_file if ipc_format == "file" else innovate.export_ipc_stream
    open_arrow = pa.ipc.open_file if ipc_format == "file" else pa.ipc.open_stream
    table = open_arrow(pa.BufferReader(to_native_ipc(frame))).read_all()
    assert table.to_pydict() == frame.to_dict(as_series=False)
    assert pa.types.is_boolean(table.schema.field("successful").type)
    assert pa.types.is_large_string(table.schema.field(json_column).type)
    if field == "comparisons":
        assert table.schema.field("comparison_index").type == pa.uint64()
    assert [json.loads(row) for row in table[json_column].to_pylist()] == rows

    # The independent reader becomes a producer; native rewriting must retain
    # its schema metadata and data when converting to the other IPC container.
    metadata = {b"innovate.qualification": b"issue-556 synthetic", b"provenance": "β".encode()}
    table = table.replace_schema_metadata(metadata)
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_file if ipc_format == "file" else pa.ipc.new_stream
    with writer(sink, table.schema) as output:
        output.write_table(table, max_chunksize=1)
    source = sink.getvalue().to_pybytes()
    to_other_ipc = innovate.export_ipc_stream if ipc_format == "file" else innovate.export_ipc_file
    open_other = pa.ipc.open_stream if ipc_format == "file" else pa.ipc.open_file
    restored = open_other(pa.BufferReader(to_other_ipc(source))).read_all()
    assert restored.schema.metadata == metadata
    assert restored.to_pydict() == table.to_pydict()
    assert [json.loads(row) for row in restored[json_column].to_pylist()] == rows
