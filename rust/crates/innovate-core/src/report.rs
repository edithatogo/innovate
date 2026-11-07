//! Historical decision-report JSON contract and claim validation.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

const SCHEMA: &str = "1.0";
const CLAIMS: [&str; 4] = ["descriptive", "predictive", "simulation", "causal"];

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Report {
    title: String,
    workflow: String,
    claims: Vec<Claim>,
    #[serde(default)]
    assumptions: Vec<String>,
    #[serde(default)]
    limitations: Vec<String>,
    #[serde(default)]
    diagnostics: Map<String, Value>,
    #[serde(default)]
    uncertainty: Map<String, Value>,
    #[serde(default)]
    sensitivity: Map<String, Value>,
    #[serde(default)]
    explainability: Map<String, Value>,
    #[serde(default)]
    metadata: Map<String, Value>,
    #[serde(default)]
    interpretation: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Claim {
    claim_type: String,
    statement: String,
    #[serde(default)]
    assumptions: Vec<String>,
    #[serde(default)]
    limitations: Vec<String>,
    #[serde(default)]
    evidence_refs: Vec<String>,
    #[serde(default)]
    allowed_as_recommendation: bool,
}

fn validate_wording(text: &str) -> Result<(), &'static str> {
    let lowered = text.to_lowercase();
    if [
        "you should implement",
        "must adopt",
        "guaranteed to",
        "clinically indicated",
        "legally required",
        "regulatory approval",
    ]
    .iter()
    .any(|phrase| lowered.contains(phrase))
    {
        return Err("public wording contains unsupported recommendation language");
    }
    Ok(())
}

fn validate_notes(notes: &mut Vec<String>) -> Result<(), &'static str> {
    for note in notes.iter_mut() {
        *note = note.trim().to_owned();
        validate_wording(note)?;
    }
    notes.retain(|note| !note.is_empty());
    Ok(())
}

pub(crate) fn export(payload: &Value) -> Result<Value, &'static str> {
    let mut report = Report::deserialize(payload).map_err(|_| "invalid decision report fields")?;
    if report.title.trim().is_empty() || report.claims.is_empty() {
        return Err("report requires a title and at least one claim");
    }
    if !matches!(
        report.workflow.as_str(),
        "policy" | "competition" | "substitution" | "diffusion" | "generic"
    ) {
        return Err("unsupported decision report workflow");
    }
    validate_wording(&report.title)?;
    validate_wording(&report.interpretation)?;
    validate_notes(&mut report.assumptions)?;
    validate_notes(&mut report.limitations)?;
    let mut counts = [0; 4];
    let mut interpretations = Vec::with_capacity(report.claims.len());
    for claim in &mut report.claims {
        claim.claim_type = claim.claim_type.trim().to_ascii_lowercase();
        let index = CLAIMS
            .iter()
            .position(|kind| *kind == claim.claim_type)
            .ok_or("unsupported decision report claim type")?;
        if claim.statement.trim().is_empty() || claim.allowed_as_recommendation {
            return Err("claim requires a statement and cannot authorize recommendations");
        }
        validate_wording(&claim.statement)?;
        validate_notes(&mut claim.assumptions)?;
        validate_notes(&mut claim.limitations)?;
        claim
            .evidence_refs
            .retain(|reference| !reference.trim().is_empty());
        if index == 3 && claim.assumptions.is_empty() {
            return Err("causal claims require an identification assumption");
        }
        counts[index] += 1;
        interpretations.push([
            "Describes observed or fitted patterns under the stated data and model. Does not imply future performance or causal effects.",
            "Projects outcomes under the fitted model and fixed assumptions. Predictions are not causal effects and are sensitive to extrapolation.",
            "Summarizes counterfactual trajectories under an explicit simulation design. Results depend on structural assumptions and scenario inputs.",
            "Claims causal interpretation only when identification assumptions are stated and supported. Sensitivity analysis does not replace identification.",
        ][index]);
    }
    if report.interpretation.trim().is_empty() {
        let dominant = counts.iter().rposition(|count| *count > 0).unwrap();
        report.interpretation = [
            "Interpret results as descriptive summaries of fitted or observed patterns. They do not authorize automated recommendations.",
            "Interpret results as model-based projections under fixed assumptions. They are not causal effects.",
            "Interpret results as simulation outcomes under explicit scenario inputs. Do not treat them as empirical policy evaluations without further design.",
            "Interpret causal statements only under the listed identification assumptions. Sensitivity results describe robustness, not proof of causality.",
        ][dominant].into();
    }
    let mut output = json!(report);
    output["schema_version"] = json!(SCHEMA);
    for (claim, interpretation) in output["claims"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .zip(interpretations)
    {
        claim["interpretation"] = json!(interpretation);
    }
    let claim_counts: Map<_, _> = CLAIMS
        .into_iter()
        .zip(counts)
        .map(|(kind, count)| (kind.to_owned(), json!(count)))
        .collect();
    output["claim_safety"] = json!({"schema_version":SCHEMA,"claim_counts":claim_counts,
        "contains_causal_claims":counts[3]>0,"recommendations_allowed":false,
        "forbidden_phrase_policy":"fail_closed"});
    Ok(output)
}
