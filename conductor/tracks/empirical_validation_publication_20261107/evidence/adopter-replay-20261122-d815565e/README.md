# Adopter-reference replay on the combined-source wheel

The macOS ARM64 wheel from [release run 45003005112](https://github.com/edithatogo/innovate/actions/runs/45003005112)
has SHA-256 `5de562362fad24ea88674d6b6e425bd0304a0c3d9faf5c724862869ba602c72b`.
Its [signed receipt](../signed-wheels-20261122-d815565e/receipt-macos-arm64.json)
binds source inventory
`188a0e3640509cfc8b61f0bc360ef76418d3be918ecc473e2c06af2e08e5765c`.
The installed `innovate/__init__.py` and native extension matched the wheel
bytes before the replay.

With Python 3.15.2 and Polars 2.1.0, the existing adopter case generator
reproduced the original case SHA-256
`aa4bb6db768641b7905bd4ccaa91ba606ab66bf083aaffc5d2efd2f499123f30`.
The public Python and Polars consumer passed 1,101 exact-reference cases, six
invalid-input cases and one fitted Logistic case. The original
[replay script and reference explanation](../adopter-classification-20261120/README.md)
remain applicable. The [recorded output](python.log) has SHA-256
`467ec1403981571b5bb9888b51e5ae6992b0ed67ee7084843afe207cbf1928c5`.

The wheel receipt is signed. This additional replay output is local execution
evidence and is not itself a signed scientific-acceptance or release manifest.
