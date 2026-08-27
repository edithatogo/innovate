import json
from innovate.causal.policy import CausalModel, InterventionContract, CausalModelContract, PolicyEvaluationError

# Test if there's any remaining vulnerability via arbitrary type instantiation
# E.g., if there's a __class__ payload or something that Pydantic / json.loads allows
malicious_payload = json.dumps({
    "intervention": {
        "name": "test",
        "timing": "post",
        "comparator": "control",
        # Attempt to inject something?
    },
    "causal_model": {
        "name": "test",
        "treatment_variable": "treatment",
        "outcome_variable": "outcome",
        "confounders": ["c1", "c2"],
        # Attempt to inject something?
    }
})

try:
    model = CausalModel.from_json(malicious_payload)
    print("Success:", model)
except Exception as e:
    print("Error:", e)
