import os
from pathlib import Path

from features import extract
from methods import evaluate

NAMES = (
    ("target", "SVM_T (3 labeled target)"),
    ("source_target", "SVM_ST (24-kernel average)"),
    ("mkl", "MKL (no MMD)"),
    ("amkl", "A-MKL"),
    ("amkl_nopre", "A-MKL without pre-learned classifiers"),
)


def main():
    root = os.environ.get("DATA_ROOT", "data")
    dists = extract(root, Path("cache") / "features.npz")
    scores = evaluate(dists)
    for key, title in NAMES:
        val = scores[key]
        mean = sum(val) / len(val)
        var = sum((v - mean) ** 2 for v in val) / (len(val) - 1)
        print(f"{title}\t{mean * 100:.1f} ± {(var ** 0.5) * 100:.1f}%")


if __name__ == "__main__":
    main()
