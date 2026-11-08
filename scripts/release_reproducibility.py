"""The 0.7 release gate remains closed until native feature migration is qualified."""
from check_migration import main

if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], "--release"]
    raise SystemExit(main())
