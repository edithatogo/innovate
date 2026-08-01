import json

import pytest

from innovate.causal.policy import CausalModel, PolicyEvaluationError


def test_causal_model_from_json_valid():
    valid_json = json.dumps(
        {
            "intervention": {"name": "test_intervention", "timing": "post", "comparator": "control"},
            "causal_model": {
                "name": "test_causal_model",
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "confounders": ["c1", "c2"],
            },
        }
    )

    model = CausalModel.from_json(valid_json)
    assert model.intervention.name == "test_intervention"
    assert model.causal_model.name == "test_causal_model"


def test_causal_model_from_json_invalid_json_string():
    invalid_json = "{"
    with pytest.raises(PolicyEvaluationError, match="Invalid JSON string"):
        CausalModel.from_json(invalid_json)


def test_causal_model_from_json_not_a_dict():
    not_a_dict_json = "[]"
    with pytest.raises(PolicyEvaluationError, match="JSON data must be a dictionary"):
        CausalModel.from_json(not_a_dict_json)


def test_causal_model_from_json_missing_intervention():
    missing_intervention = json.dumps(
        {
            "causal_model": {
                "name": "test_causal_model",
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "confounders": ["c1", "c2"],
            }
        }
    )
    with pytest.raises(PolicyEvaluationError, match="Missing 'intervention' key in JSON data"):
        CausalModel.from_json(missing_intervention)


def test_causal_model_from_json_intervention_not_dict():
    intervention_not_dict = json.dumps(
        {
            "intervention": "not_a_dict",
            "causal_model": {
                "name": "test_causal_model",
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "confounders": ["c1", "c2"],
            },
        }
    )
    with pytest.raises(PolicyEvaluationError, match="'intervention' must be a dictionary"):
        CausalModel.from_json(intervention_not_dict)


def test_causal_model_from_json_missing_causal_model():
    missing_causal_model = json.dumps(
        {"intervention": {"name": "test_intervention", "timing": "post", "comparator": "control"}}
    )
    with pytest.raises(PolicyEvaluationError, match="Missing 'causal_model' key in JSON data"):
        CausalModel.from_json(missing_causal_model)


def test_causal_model_from_json_causal_model_not_dict():
    causal_model_not_dict = json.dumps(
        {
            "intervention": {"name": "test_intervention", "timing": "post", "comparator": "control"},
            "causal_model": "not_a_dict",
        }
    )
    with pytest.raises(PolicyEvaluationError, match="'causal_model' must be a dictionary"):
        CausalModel.from_json(causal_model_not_dict)


def test_causal_model_from_json_extra_keys_rejected():
    invalid_json = json.dumps(
        {
            "intervention": {
                "name": "test_intervention",
                "timing": "post",
                "comparator": "control",
                "malicious_key": "injected_value",
            },
            "causal_model": {
                "name": "test_causal_model",
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "confounders": ["c1", "c2"],
            },
        }
    )
    with pytest.raises(PolicyEvaluationError, match="Unknown fields in 'intervention': malicious_key"):
        CausalModel.from_json(invalid_json)


def test_causal_model_from_json_missing_keys_rejected():
    invalid_json = json.dumps(
        {
            "intervention": {
                "timing": "post",
                "comparator": "control",
            },
            "causal_model": {
                "name": "test_causal_model",
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
                "confounders": ["c1", "c2"],
            },
        }
    )
    # Missing "name" in intervention contract
    with pytest.raises(PolicyEvaluationError, match="Data validation error"):
        CausalModel.from_json(invalid_json)
