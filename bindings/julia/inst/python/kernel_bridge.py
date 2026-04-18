"""Placeholder Python entrypoint for Julia bindings.

The Julia package uses this bridge path as the contract boundary for future
kernel calls. The actual runtime bridge will be wired in later phases.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit("Julia kernel bridge scaffold not yet implemented")


if __name__ == "__main__":
    main()
