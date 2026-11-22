# Hosted offline security evidence for the combined source

[PR run 45015728697](https://github.com/edithatogo/innovate/actions/runs/45015728697)
passed on commit `1f45e1622eddf705596f586856f82cd79d1bc946` using policy source
`7d8aa3fc88f86cc0c54b2c4d57ca9a4bfa97bdda`. Artifact `14752227048`
matched archive SHA-256 `5e03d651e5a43596f17cfebe1de318d06147a9894361fa9f81e42d94b1ae96a6`.

The retained reports show all eight image qualification controls passed,
the scanner had no network access, and all eight image findings had verified
backports with zero ignored findings. They do not qualify runner registration,
application adapters, image redistribution or release acceptance.

The separate push-triggered run initially failed on an HTTP 503 after its
bounded download retries; its second attempt passed. The interrupted local
x86-64 replay remains partial and is recorded separately.
