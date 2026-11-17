# [AI-SCAFFOLD] Pipeline fixes implemented with AI assistance; include in course disclosure.
"""Execute extracted notebook sections in order, using explicit module dependencies."""

import argparse
from importlib import import_module
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, choices=range(1, 15), default=14,
                        help="Last section to execute (default: 14).")
    parser.add_argument("--evaluate", choices=["baselines", "neural", "ablations"],
                        help="Execute shared evaluation instead of importing sections.")
    args, extra = parser.parse_known_args()
    if args.evaluate:
        from evaluation import main as evaluate
        flags = [] if args.evaluate == "baselines" else ["--neural"]
        if args.evaluate == "ablations":
            flags += ["--ablations", "--require-all-families"]
        return evaluate(flags + extra)
    if extra:
        parser.error("Evaluation flags require --evaluate")
    section_dir = Path(__file__).resolve().parent
    for path in sorted(section_dir.glob("[0-9][0-9]*.py")):
        if int(path.name[:2]) > args.through:
            break
        print(f"\nRunning {path.name}")
        import_module(path.stem)


if __name__ == "__main__":
    main()
