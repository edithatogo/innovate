//! Typed inference status, fail-closed posterior access, and analytic log-density contracts.

pub mod hype_fit;
pub mod hype_posterior;
pub mod status;

pub use status::{
    DeterministicFitResult, FitFailure, FitOutcome, FitResult, InferenceSession, InnovationError, ObservationSeries,
    PosteriorDraws, log_density, posterior,
};
