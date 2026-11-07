"""Replace build-root URIs in an unsigned wheel's SBOM and refresh RECORD."""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile


def normalize(wheel: Path, source_root: Path, output: Path) -> None:
    if wheel.resolve() == output.resolve():
        raise ValueError("output must be separate from the original wheel")
    prefix = source_root.resolve().as_uri() + "/"
    with ZipFile(wheel) as source:
        names = source.namelist()
        records = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(records) != 1 or len(names) != len(set(names)):
            raise ValueError("expected one RECORD and unique wheel members")
        record = records[0]
        if any(record + suffix in names for suffix in (".jws", ".p7s")):
            raise ValueError("expected an unsigned wheel")
        changes = {}
        for name in names:
            if ".dist-info/sboms/" in name and name.endswith(".cyclonedx.json"):
                raw = source.read(name).decode("utf-8")
                json.loads(raw)
                updated = raw.replace(prefix, "file:///source/")
                if updated != raw:
                    changes[name] = updated.encode("utf-8")
        if not changes:
            raise ValueError("no matching source-root references in wheel SBOMs")
        rows = list(csv.reader(io.StringIO(source.read(record).decode("utf-8"))))
        for row in rows:
            if row[0] in changes:
                data = changes[row[0]]
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
                row[1:] = [f"sha256={digest}", str(len(data))]
        if not changes.keys() <= {row[0] for row in rows}:
            raise ValueError("SBOM is missing from RECORD")
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        changes[record] = buffer.getvalue().encode("utf-8")
        with ZipFile(output, "x") as target:
            for info in source.infolist():
                target.writestr(info, changes[info.filename] if info.filename in changes else source.read(info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    normalize(args.wheel, args.source_root, args.output)
