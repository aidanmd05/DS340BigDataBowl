"""Execute extracted notebook sections in order, using explicit module dependencies."""

import argparse
from importlib import import_module
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, choices=range(1, 15), default=14,
                        help="Last section to execute (default: 14).")
    args = parser.parse_args()
    section_dir = Path(__file__).resolve().parent
    for path in sorted(section_dir.glob("[0-9][0-9]*.py")):
        if int(path.name[:2]) > args.through:
            break
        print(f"\nRunning {path.name}")
        import_module(path.stem)


if __name__ == "__main__":
    main()
