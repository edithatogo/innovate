# Voiage Diffusion-Uncertainty Fixture v1

This documented-stage fixture gives `voiage` examples a deterministic
diffusion-uncertainty source without importing `innovate` internals or `voiage`
runtime objects.

Files:

- `manifest.json`: schema metadata, VOI concept mappings, provenance,
  promotion-stage notes, and stable sample dimensions.
- `parameter_draws.csv`: joint parameter draws keyed by `scenario_id`,
  `draw_id`, and `parameter_name`.
- `adoption_trajectories.csv`: deterministic adoption trajectories keyed by
  `scenario_id`, `draw_id`, and `time`.

The CSV payloads use Arrow-compatible logical types documented in the manifest.
Production adapters should preserve the column names and logical types when
promoting the payload to Arrow or Parquet.

Out of scope:

- implementing EVPI, EVPPI, EVSI, ENBS, or other VOI methods
- adding `voiage` as an `innovate` dependency
- shipping a runtime adapter
- declaring a supported compatibility matrix
