import numpy as np
from sklearn.svm import SVC

CLASSES = ("animal", "water", "city", "vehicle")
FEATURES = ("sift", "hof", "color")
LEVELS = ("L0", "L1")
GAMMAS = (0.25, 0.5, 1.0, 2.0)
C = 1.0
N_LABELED = 3
N_SEEDS = 10


def psd(kernel):
    kernel = 0.5 * (kernel + kernel.T)
    w, v = np.linalg.eigh(kernel)
    return (v * np.clip(w, 0, None)) @ v.T


def base_kernels(dists):
    kernels, names = [], []
    for feature in FEATURES:
        for level in LEVELS:
            dist = dists[f"{feature}__{level}"]
            scale = dist[np.triu_indices(len(dist), 1)].mean()
            for gamma in GAMMAS:
                kernels.append(psd(np.exp(-gamma * dist / scale)))
                names.append(f"{feature}_{level}_g{gamma}")
    return kernels, names


def mmd(kernel, src, tgt):
    return float(
        kernel[np.ix_(src, src)].mean()
        + kernel[np.ix_(tgt, tgt)].mean()
        - 2 * kernel[np.ix_(src, tgt)].mean()
    )


def fit_ovr(k_train, y):
    models = []
    for c in range(len(CLASSES)):
        yb = np.where(y == c, 1, -1)
        model = SVC(C=C, kernel="precomputed", class_weight="balanced", tol=1e-4)
        model.fit(k_train, yb)
        models.append(model)
    return models


def predict(models, k_test):
    scores = [model.decision_function(k_test) for model in models]
    return np.argmax(np.stack(scores, 1), 1)


def alpha_y(models, n):
    rows = []
    for model in models:
        ay = np.zeros(n)
        ay[model.support_] = model.dual_coef_[0]
        rows.append(ay)
    return rows


def prelearned(kernels, names, src, y):
    n = len(y)
    outs = []
    for kernel, name in zip(kernels, names):
        if not name.endswith("_g1.0"):
            continue
        models = fit_ovr(kernel[np.ix_(src, src)], y[src])
        cols = [model.decision_function(kernel[np.ix_(np.arange(n), src)]) for model in models]
        outs.append(np.stack(cols, 1))
    return np.stack(outs, 2)


def svm_fixed(kernel, train, test, y):
    models = fit_ovr(kernel[np.ix_(train, train)], y[train])
    return predict(models, kernel[np.ix_(test, train)])


def proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1.0), 0)


def amkl(kernels, src, tgt, train, test, y, theta, use_mmd, use_pre, pre=None, iters=30):
    m = len(kernels)
    if use_mmd:
        p = np.array([mmd(k, src, tgt) for k in kernels])
        p = p / (np.abs(p).max() + 1e-12)
    else:
        p = np.zeros(m)
    k_train = [k[np.ix_(train, train)] for k in kernels]
    k_test = [k[np.ix_(test, train)] for k in kernels]
    if use_pre:
        f_train, f_test = pre[train], pre[test]

    def extra(c):
        if not use_pre:
            return 0.0, 0.0
        n_pre = f_train.shape[2]
        return f_train[:, c, :] @ f_train[:, c, :].T / n_pre, f_test[:, c, :] @ f_train[:, c, :].T / n_pre

    def fit(weights):
        models = []
        for c in range(len(CLASSES)):
            kc = sum(weights[i] * k_train[i] for i in range(m))
            e_train, _ = extra(c)
            yb = np.where(y[train] == c, 1, -1)
            model = SVC(C=C, kernel="precomputed", class_weight="balanced", tol=1e-4)
            model.fit(kc + e_train, yb)
            models.append(model)
        return models

    weights = np.ones(m) / m
    g_scale = None
    for step_id in range(iters):
        models = fit(weights)
        grad_j = np.zeros(m)
        for ay in alpha_y(models, len(train)):
            for i in range(m):
                grad_j[i] -= 0.5 * ay @ k_train[i] @ ay
        if g_scale is None:
            g_scale = np.abs(grad_j).max() + 1e-12
        grad = (weights @ p) * p + theta * grad_j / g_scale
        step = 0.5 / (1 + step_id) / (np.abs(grad).max() + 1e-12)
        weights = proj_simplex(weights - step * grad)
    models = fit(weights)
    scores = []
    for c, model in enumerate(models):
        kc = sum(weights[i] * k_test[i] for i in range(m))
        _, e_test = extra(c)
        scores.append(model.decision_function(kc + e_test))
    return np.argmax(np.stack(scores, 1), 1), weights.copy()


def splits(y, tgt):
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        labeled, test = [], []
        for c in range(len(CLASSES)):
            ids = tgt[y[tgt] == c].copy()
            rng.shuffle(ids)
            labeled += ids[:N_LABELED].tolist()
            test += ids[N_LABELED:].tolist()
        yield np.array(labeled), np.array(test)


def evaluate(dists):
    kernels, names = base_kernels(dists)
    y = np.array([CLASSES.index(label) for label in dists["labels"]])
    src = np.where(dists["domains"] == "source")[0]
    tgt = np.where(dists["domains"] == "target")[0]
    for c, name in enumerate(CLASSES):
        if int((y[tgt] == c).sum()) <= N_LABELED:
            raise RuntimeError(f"target class {name} needs more than {N_LABELED} clips")
    average = sum(kernels) / len(kernels)
    pre = prelearned(kernels, names, src, y)
    keys = ("target", "source_target", "mkl", "amkl", "amkl_nopre")
    scores = {name: [] for name in keys}
    correct = {name: [] for name in keys}
    learned = {name: [] for name in ("mkl", "amkl", "amkl_nopre")}
    n_test = None
    for labeled, test in splits(y, tgt):
        train = np.concatenate([src, labeled])
        yt = y[test]
        n_test = int(len(test))
        pred = svm_fixed(average, labeled, test, y)
        hit = int((pred == yt).sum())
        correct["target"].append(hit)
        scores["target"].append(hit / n_test)
        pred = svm_fixed(average, train, test, y)
        hit = int((pred == yt).sum())
        correct["source_target"].append(hit)
        scores["source_target"].append(hit / n_test)
        for key, use_mmd, use_pre in (
            ("mkl", False, False),
            ("amkl", True, True),
            ("amkl_nopre", True, False),
        ):
            pred, weights = amkl(
                kernels, src, tgt, train, test, y, 1.0, use_mmd, use_pre, pre if use_pre else None
            )
            hit = int((pred == yt).sum())
            correct[key].append(hit)
            scores[key].append(hit / n_test)
            learned[key].append(weights)
    return {
        "accuracy": scores,
        "correct": correct,
        "n_test": n_test,
        "kernel_names": names,
        "weights": {key: np.stack(rows) for key, rows in learned.items()},
    }
