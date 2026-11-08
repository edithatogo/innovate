# Native Bass dispatch contract

The root workspace implements Bass simulation through
`innovate_core::KernelBinding::simulate_model` and `innovate_core::execute_json`.
Send schema `2.0`, operation `simulate_model`, model key `bass`, and
simulation options selecting `kairos_des`, `kairos_abm` or `kairos_network`.
Rust validates the request and owns the scientific model, event dispatch
and random streams. Host bindings marshal requests and results.

The Python wheel calls the same core through its PyO3 extension. Julia
and R call `innovate_call` with borrowed, NUL-terminated UTF-8 and release
the returned Rust allocation exactly once with `innovate_free`.
A null response pointer indicates no string was returned. A normal
response is a JSON envelope with schema version, result and error.
Invalid model inputs and unsupported engines return an error instead
of executing an alternative model.

The simulation result includes the sampled values and counts. Its
metadata identifies the engine, pinned Kairos revision, effective seed
and number of dispatched events. The count can be zero for a valid
run that requires no events. RNG checkpoint data records provenance.
The current contract supports replay from the request and seed.

## Separate verification leaves

- Bridge boundary. Exercise a valid Bass request and an unsupported
  engine through the Rust C ABI, checking schema and error shape and
  releasing both owned responses. Scope the change to the FFI crate.
- Installed consumers. Send one identical seeded Bass request through
  the built Python wheel, Julia and installed R. Compare the sampled
  counts and dispatch metadata and check unsupported-engine rejection.
  Scope durable test changes to the three binding suites.

## Verification on the migrated workspace

- The locked FFI regression at `30afd938` sends both requests through
  `innovate_call`. The seeded DES request returns counts [0, 1] and
  one dispatch. The unknown engine returns an error and a null result.
  Both response allocations are released.
- The installed-consumer check at `ae78919b` exercised the same
  requests through Python 3.15.2, Julia and R against runtime source
  `f73933fe`. All three reproduced the counts and dispatch and raised
  the native invalid-request error for the unknown engine.
- The scheduler tie/cancel fixture at `0d664bdb` and ABM state and
  random-stream fixture at `0afbb148` run against the locked root
  workspace. These are local runtime checks. The imported track
  retains its formal dependency and integration gates.
