"""V03 cross-domain cohort semantics and early-stage validation tests."""

from __future__ import annotations

import json
from pathlib import Path


def test_cross_domain_cohorts_preserve_observation_semantics():
    protocol = json.loads(Path("validation/cross-domain/protocol.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in protocol["cohorts"]}

    assert by_id["pubmed_attention"]["observation_kind"] == "Attention"
    assert by_id["pubmed_attention"]["allow_relabel_as_new_adopters"] is False
    assert by_id["chat_cer_ev"]["observation_kind"] == "Share"
    assert by_id["ibm_dimora_tech"]["observation_kind"] == "NewAdoption"
