import base64
import csv
import hashlib
import io
import json
from zipfile import ZipFile, ZipInfo

import pytest

from scripts.normalize_wheel_sbom import normalize

RECORD = "example.dist-info/RECORD"
SBOM = "example.dist-info/sboms/example.cyclonedx.json"


def make_wheel(path, root, *, signed=False, corrupt=False):
    with ZipFile(path, "w") as wheel:
        payload = json.dumps({"bom-ref": root.as_uri() + "/Cargo.toml"}).encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        wheel.writestr(ZipInfo(SBOM), payload + (b" " if corrupt else b""))
        wheel.writestr(ZipInfo(RECORD), f"{SBOM},sha256={digest},{len(payload)}\n{RECORD},,\n")
        if signed:
            wheel.writestr(ZipInfo(RECORD + ".jws"), "signature")


def test_distinct_source_roots_produce_identical_wheels(tmp_path):
    outputs = []
    for name in ("a", "b"):
        root = tmp_path / name
        original = tmp_path / f"{name}.whl"
        output = tmp_path / f"{name}-normalized.whl"
        make_wheel(original, root)
        normalize(original, root, output)
        outputs.append(output.read_bytes())
        with ZipFile(output) as wheel:
            raw = wheel.read(SBOM)
            assert json.loads(raw)["bom-ref"] == "file:///source/Cargo.toml"
            digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
            assert list(csv.reader(io.StringIO(wheel.read(RECORD).decode()))) == [
                [SBOM, f"sha256={digest}", str(len(raw))],
                [RECORD, "", ""],
            ]
    assert outputs[0] == outputs[1]


def test_signed_wheel_is_rejected_without_writing(tmp_path):
    original = tmp_path / "signed.whl"
    output = tmp_path / "normalized.whl"
    make_wheel(original, tmp_path, signed=True)
    with pytest.raises(ValueError, match="unsigned wheel"):
        normalize(original, tmp_path, output)
    assert not output.exists()


def test_corrupt_payload_is_rejected_without_writing(tmp_path):
    original = tmp_path / "corrupt.whl"
    output = tmp_path / "normalized.whl"
    make_wheel(original, tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="invalid RECORD digest or size"):
        normalize(original, tmp_path, output)
    assert not output.exists()
