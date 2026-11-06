import json
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


def sample_std(values):
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def write_results(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(report["kernel_names"])
    methods = {}
    for key, title in NAMES:
        mean, std = sample_std(report["accuracy"][key])
        methods[title] = {
            "mean": mean,
            "std": std,
            "correct": report["correct"][key],
            "n_test": report["n_test"],
        }
        print(f"{title}\t{mean * 100:.1f} ± {std * 100:.1f}%")
    weights = {}
    for key, title in NAMES:
        if key not in report["weights"]:
            continue
        mean_w = report["weights"][key].mean(0)
        weights[title] = {name: float(value) for name, value in zip(names, mean_w)}
    payload = {"n_test": report["n_test"], "n_seeds": len(report["correct"]["target"]), "methods": methods, "kernel_weights": weights}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.as_posix()}")


def main():
    raw = os.environ.get("RAW_ROOT")
    root = os.environ.get("DATA_ROOT", "data")
    if raw:
        from prepare import prepare

        prepare(raw, root)
    dists = extract(root, Path("cache") / "features.npz")
    report = evaluate(dists)
    write_results(report, Path("results") / "metrics.json")


if __name__ == "__main__":
    main()
