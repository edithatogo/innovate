"""Copy the actual publication wheel's embedded Rust inventory for Grype."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile


def extract(wheel: Path, output: Path) -> None:
    """Require one complete embedded inventory and leave publication bytes intact."""
    with ZipFile(wheel) as archive:
        entries = [
            entry
            for entry in archive.infolist()
            if ".dist-info/sboms/" in entry.filename and entry.filename.endswith(".cyclonedx.json")
        ]
        if len(entries) != 1 or entries[0].file_size > 20 * 1024 * 1024:
            raise ValueError("expected one bounded embedded Rust CycloneDX inventory")
        raw = archive.read(entries[0])
    inventory = json.loads(raw)
    components = inventory.get("components", [])
    if inventory.get("bomFormat") != "CycloneDX" or not any(
        isinstance(component.get("purl"), str) and component["purl"].startswith("pkg:cargo/")
        for component in components
    ):
        raise ValueError("embedded wheel inventory has no resolved Rust packages")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.wheel, args.output)
