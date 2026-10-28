"""Run the full validation study.

Usage:
    python run_validation.py --data data/creditcard.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fraudval.pipeline import run, update_readme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/creditcard.csv", help="path to creditcard.csv")
    ap.add_argument("--out", default="reports", help="output folder for the report and figures")
    ap.add_argument("--bootstrap", type=int, default=500, help="bootstrap resamples for confidence intervals")
    args = ap.parse_args()

    if not Path(args.data).exists():
        sys.exit(f"Data file not found: {args.data}. See data/README.md for how to download it.")

    res = run(args.data, args.out, n_boot=args.bootstrap)
    update_readme(res, Path(__file__).parent / "README.md")
    print(res["comparison"][["model", "valid_pr_auc", "test_pr_auc", "test_recall", "test_precision"]].round(3).to_string(index=False))
    print(f"\nSelected: {res['champion']}  |  Overall rating: {res['overall']}  |  Report: {args.out}/validation_report.md")


if __name__ == "__main__":
    main()
