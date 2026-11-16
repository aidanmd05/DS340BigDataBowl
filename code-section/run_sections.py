"""Execute extracted notebook sections in order, sharing their variables."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, choices=range(1, 15), default=14,
                        help="Last section to execute (default: 14).")
    args = parser.parse_args()
    section_dir = Path(__file__).resolve().parent
    namespace = {"__name__": "__main__", "display": print}
    for path in sorted(section_dir.glob("[0-9][0-9]*.py")):
        if int(path.name[:2]) > args.through:
            break
        print(f"\nRunning {path.name}")
        namespace["__file__"] = str(path)
        exec(compile(path.read_text(), str(path), "exec"), namespace)


if __name__ == "__main__":
    main()
