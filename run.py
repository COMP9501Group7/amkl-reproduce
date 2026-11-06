import json
import os
from pathlib import Path

from features import extract
from methods import CLASSES, evaluate

NAMES = (
    ("svm_s", "SVM_S"),
    ("svm_t", "SVM_T"),
    ("svm_st", "SVM_ST"),
    ("fr", "FR"),
    ("mkl", "MKL"),
    ("amkl", "A-MKL"),
    ("amkl_nopre", "A-MKL_noPre"),
)


def sample_std(values):
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def pack(values):
    mean, std = sample_std(values)
    return {"mean": mean, "std": std, "per_seed": [float(v) for v in values]}


def write_results(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(report["kernel_names"])
    methods = {}
    print("method\taccuracy\tmacro_precision")
    for key, title in NAMES:
        acc = pack(report["accuracy"][key])
        prec = pack(report["precision"][key])
        by_class = {}
        columns = list(zip(*report["precision_per_class"][key]))
        for label, column in zip(CLASSES, columns):
            by_class[label] = pack(column)
        methods[title] = {
            "accuracy": acc,
            "macro_precision": prec,
            "precision_per_class": by_class,
            "correct": report["correct"][key],
            "n_test": report["n_test"],
        }
        print(f"{title}\t{acc['mean'] * 100:.1f} ± {acc['std'] * 100:.1f}%\t{prec['mean'] * 100:.1f} ± {prec['std'] * 100:.1f}%")
    weights = {}
    for key, title in NAMES:
        if key not in report["weights"]:
            continue
        mean_w = report["weights"][key].mean(0)
        weights[title] = {name: float(value) for name, value in zip(names, mean_w)}
    payload = {
        "n_test": report["n_test"],
        "n_seeds": len(report["correct"]["svm_t"]),
        "methods": methods,
        "kernel_weights": weights,
    }
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
