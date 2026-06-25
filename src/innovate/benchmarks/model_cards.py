"""Model-card schema and registry for stable innovate model families."""

from __future__ import annotations

from dataclasses import dataclass, field

from innovate.capabilities import ModelCapability, get_model_registry


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Machine-readable description of a stable model family's contract."""

    model_key: str
    model_name: str
    family: str
    stability: str
    summary: str
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    inputs: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    benchmark_case_ids: tuple[str, ...] = field(default_factory=tuple)
    supported_backends: tuple[str, ...] = field(default_factory=tuple)
    import_path: str = ""

    def __post_init__(self) -> None:
        """Validate required model-card fields."""
        if not self.model_key:
            raise ValueError("Model card model_key must be non-empty.")
        if not self.model_name:
            raise ValueError("Model card model_name must be non-empty.")
        if not self.summary:
            raise ValueError("Model card summary must be non-empty.")
        if not self.assumptions:
            raise ValueError("Model card assumptions must be non-empty.")
        if not self.inputs:
            raise ValueError("Model card inputs must be non-empty.")
        if not self.outputs:
            raise ValueError("Model card outputs must be non-empty.")
        if not self.diagnostics:
            raise ValueError("Model card diagnostics must be non-empty.")
        if not self.limitations:
            raise ValueError("Model card limitations must be non-empty.")
        if not self.benchmark_case_ids:
            raise ValueError("Model card benchmark_case_ids must be non-empty.")

    def validate(self) -> None:
        """Re-run validation explicitly for callers that want an imperative check."""
        self.__post_init__()

    def to_dict(self) -> dict[str, object]:
        """Serialize the card into a JSON-friendly dictionary."""
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "family": self.family,
            "stability": self.stability,
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "diagnostics": list(self.diagnostics),
            "limitations": list(self.limitations),
            "benchmark_case_ids": list(self.benchmark_case_ids),
            "supported_backends": list(self.supported_backends),
            "import_path": self.import_path,
        }


def _build_model_card(capability: ModelCapability) -> ModelCard:
    """Construct a model card from stable capability metadata."""
    blueprint = _MODEL_CARD_BLUEPRINTS[capability.key]
    return ModelCard(
        model_key=capability.key,
        model_name=blueprint["model_name"],
        family=capability.family,
        stability=capability.stability,
        summary=blueprint["summary"],
        assumptions=blueprint["assumptions"],
        inputs=blueprint["inputs"],
        outputs=blueprint["outputs"],
        diagnostics=blueprint["diagnostics"],
        limitations=blueprint["limitations"],
        benchmark_case_ids=blueprint["benchmark_case_ids"],
        supported_backends=capability.supported_backends,
        import_path=capability.import_path,
    )


_MODEL_CARD_BLUEPRINTS: dict[str, dict[str, tuple[str, ...] | str]] = {
    "lock_in": {
        "model_name": "LockInModel",
        "summary": "Path dependence and lock-in between competing technologies with network effects.",
        "assumptions": (
            "Intrinsic growth rates and network effects are constant.",
            "Negative influence from competitor is proportional to its adoption.",
        ),
        "inputs": ("time", "observed adoption share for each technology"),
        "outputs": ("fitted parameters", "predicted adoption trajectories", "equilibrium shares"),
        "diagnostics": ("r2", "rmse"),
        "limitations": ("Simplified two-technology model; does not capture multi-technology dynamics.",),
        "benchmark_case_ids": ("lock_in_smoke",),
    },
    "bass": {
        "model_name": "BassModel",
        "summary": "Core diffusion baseline for cumulative adoption curves.",
        "assumptions": (
            "Adoption is cumulative and monotonic.",
            "Innovation and imitation effects are sufficient to explain uptake.",
            "The benchmark corpus is synthetic and reproducible.",
        ),
        "inputs": ("time", "observed cumulative adoption"),
        "outputs": ("fitted parameters", "predicted cumulative adoption", "fit metrics"),
        "diagnostics": ("r2", "rmse", "residual_analysis"),
        "limitations": ("Does not capture network structure or policy shocks.",),
        "benchmark_case_ids": ("bass_smoke_adoption",),
    },
    "logistic": {
        "model_name": "LogisticModel",
        "summary": "Smooth S-curve baseline for adoption and growth comparison.",
        "assumptions": (
            "The underlying process follows a single smooth inflection.",
            "Observed values are reproducible and monotonic.",
            "Synthetic benchmark cases provide deterministic comparisons.",
        ),
        "inputs": ("time", "observed cumulative adoption"),
        "outputs": ("fitted parameters", "predicted cumulative adoption", "fit metrics"),
        "diagnostics": ("r2", "rmse", "residual_analysis"),
        "limitations": ("Not designed to represent network or policy mechanisms.",),
        "benchmark_case_ids": ("logistic_growth_smoke",),
    },
    "gompertz": {
        "model_name": "GompertzModel",
        "summary": "Asymmetric growth baseline for adoption curves with early acceleration.",
        "assumptions": (
            "Adoption growth is asymmetric around the inflection point.",
            "Synthetic observations are stable across runs.",
            "The benchmark corpus is intended for reproducible smoke tests.",
        ),
        "inputs": ("time", "observed cumulative adoption"),
        "outputs": ("fitted parameters", "predicted cumulative adoption", "fit metrics"),
        "diagnostics": ("r2", "rmse", "residual_analysis"),
        "limitations": ("Does not encode exogenous network or policy effects.",),
        "benchmark_case_ids": ("logistic_growth_smoke",),
    },
    "fisher_pry": {
        "model_name": "FisherPryModel",
        "summary": "Replacement-share baseline for substitution dynamics.",
        "assumptions": (
            "The benchmark target is a bounded share process.",
            "Substitution is represented as a smooth transition.",
            "Synthetic data is used for reproducibility.",
        ),
        "inputs": ("time", "observed market share"),
        "outputs": ("fitted parameters", "predicted market share", "fit metrics"),
        "diagnostics": ("r2", "rmse", "residual_analysis"),
        "limitations": ("Not a structural model of competition or policy diffusion.",),
        "benchmark_case_ids": ("fisher_pry_replacement_smoke",),
    },
    "norton_bass": {
        "model_name": "NortonBassModel",
        "summary": "Substitution model for multi-phase replacement and diffusion.",
        "assumptions": (
            "Replacement follows a structured cumulative share process.",
            "Synthetic benchmark cases remain deterministic.",
            "The model is evaluated on reproducible substitution scenarios.",
        ),
        "inputs": ("time", "observed market share"),
        "outputs": ("fitted parameters", "predicted market share", "fit metrics"),
        "diagnostics": ("r2", "rmse", "residual_analysis"),
        "limitations": ("Does not explicitly model network topology.",),
        "benchmark_case_ids": ("fisher_pry_replacement_smoke",),
    },
    "composite": {
        "model_name": "CompositeDiffusionModel",
        "summary": "Composable substitution baseline for stable diffusion comparisons.",
        "assumptions": (
            "Component diffusion processes can be combined reproducibly.",
            "Synthetic cases are used to compare stable outputs.",
            "Benchmark behavior should be deterministic across runs.",
        ),
        "inputs": ("time", "observed market share"),
        "outputs": ("fitted parameters", "predicted market share", "fit metrics"),
        "diagnostics": ("r2", "rmse", "residual_analysis"),
        "limitations": ("Representation is limited to the benchmarked substitution proxy.",),
        "benchmark_case_ids": ("fisher_pry_replacement_smoke",),
    },
    "multi_product": {
        "model_name": "MultiProductDiffusionModel",
        "summary": "Core competition model for comparing multiple product trajectories.",
        "assumptions": (
            "The focal competition process is deterministic in the benchmark corpus.",
            "Stable outputs are computed from reproducible synthetic cases.",
            "Multi-product comparisons can be summarized consistently.",
        ),
        "inputs": ("time", "observed focal share"),
        "outputs": ("fitted parameters", "predicted focal share", "fit metrics"),
        "diagnostics": ("r2", "rmse", "residual_analysis"),
        "limitations": ("Does not capture unobserved strategic shocks.",),
        "benchmark_case_ids": ("lotka_volterra_competition_smoke",),
    },
    "lotka_volterra": {
        "model_name": "LotkaVolterraModel",
        "summary": "Competition baseline for interacting adoption trajectories.",
        "assumptions": (
            "Competition can be summarized by a stable focal-share benchmark.",
            "Synthetic cases are reproducible and deterministic.",
            "The benchmark corpus emphasizes comparable outputs.",
        ),
        "inputs": ("time", "observed focal share"),
        "outputs": ("fitted parameters", "predicted focal share", "fit metrics"),
        "diagnostics": ("r2", "rmse", "residual_analysis"),
        "limitations": ("Not a full structural market simulation.",),
        "benchmark_case_ids": ("lotka_volterra_competition_smoke",),
    },
    "complementary_goods": {
        "model_name": "ComplementaryGoodsModel",
        "summary": "Stable ecosystem model for interdependent adoption trajectories.",
        "assumptions": (
            "Complementarity effects are evaluated on synthetic deterministic cases.",
            "Benchmark outputs should remain comparable across runs.",
            "The model family exposes stable summary fields.",
        ),
        "inputs": ("time", "observed focal share"),
        "outputs": ("fitted parameters", "predicted focal share", "fit metrics"),
        "diagnostics": ("r2", "rmse", "residual_analysis"),
        "limitations": ("Not all ecosystem effects are represented in the benchmark corpus.",),
        "benchmark_case_ids": ("lotka_volterra_competition_smoke",),
    },
}


def list_model_cards() -> dict[str, ModelCard]:
    """Return synchronized model cards for all stable model capabilities."""
    registry = get_model_registry()
    stable_cards = {
        key: _build_model_card(capability) for key, capability in registry.items() if capability.stability == "stable"
    }
    return dict(sorted(stable_cards.items(), key=lambda item: item[0]))


def get_model_card(model_key: str) -> ModelCard:
    """Return a synchronized model card for a stable model family."""
    cards = list_model_cards()
    try:
        return cards[model_key]
    except KeyError as exc:
        raise KeyError(f"Unknown model card: {model_key}") from exc
