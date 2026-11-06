import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

cv2.setNumThreads(1)

CLASSES = ("animal", "water", "city", "vehicle")
N_FRAMES = 8
WIDTH = 160
SIFT_K = 64
HOF_K = 32
COLOR_K = 48


def list_clips(root):
    root = Path(root)
    items = []
    for domain in ("source", "target"):
        for label in CLASSES:
            paths = sorted((root / domain / label).glob("*.mp4"))
            if not paths:
                raise FileNotFoundError(f"no clips in {domain}/{label}")
            for p in paths:
                items.append((domain, label, p))
    return items


def resize(frame, width=WIDTH):
    h, w = frame.shape[:2]
    nh = max(1, int(round(h * (width / float(w)))))
    return cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)


def sample_frames(path, k=N_FRAMES):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if 1 < n < 100000:
        for i in np.linspace(0, n - 1, k).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(resize(frame))
    cap.release()
    if len(frames) < 4:
        cap = cv2.VideoCapture(str(path))
        buf = []
        while len(buf) < 240:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            buf.append(resize(frame))
        cap.release()
        if buf:
            pick = np.linspace(0, len(buf) - 1, min(k, len(buf))).astype(int)
            frames = [buf[i] for i in pick]
    if len(frames) < 2:
        raise RuntimeError(f"unreadable clip: {path.name}")
    return frames


def sift_of(frames):
    sift = cv2.SIFT_create(nfeatures=40)
    desc, xy, tt = [], [], []
    for t, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, raw = sift.detectAndCompute(gray, None)
        if raw is None:
            continue
        h, w = gray.shape
        for p, row in zip(kp, raw):
            desc.append(np.sqrt(row.astype(np.float64) / (row.sum() + 1e-8)))
            xy.append((p.pt[0] / w, p.pt[1] / h))
            tt.append(t)
    if not desc:
        return np.zeros((0, 128)), np.zeros((0, 2)), np.zeros(0, np.int32)
    return np.stack(desc), np.asarray(xy), np.asarray(tt, np.int32)


def hof_of(frames):
    desc, xy, tt = [], [], []
    for t in range(len(frames) - 1):
        g1 = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[t + 1], cv2.COLOR_BGR2GRAY)
        h, w = g1.shape
        nh = max(1, int(round(h * 96 / w)))
        g1 = cv2.resize(g1, (96, nh))
        g2 = cv2.resize(g2, (96, nh))
        h, w = g1.shape
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 2, 12, 2, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        for r in range(4):
            for c in range(4):
                ys, ye = r * h // 4, (r + 1) * h // 4
                xs, xe = c * w // 4, (c + 1) * w // 4
                cell_m, cell_a = mag[ys:ye, xs:xe], ang[ys:ye, xs:xe]
                if cell_m.size == 0:
                    continue
                hist, _ = np.histogram(cell_a, bins=np.linspace(0, 2 * np.pi, 9), weights=cell_m)
                hist = hist / (hist.sum() + 1e-8)
                desc.append(np.concatenate([hist, [np.log1p(float(cell_m.mean()))]]))
                xy.append(((c + 0.5) / 4, (r + 0.5) / 4))
                tt.append(t)
    if not desc:
        return np.zeros((0, 9)), np.zeros((0, 2)), np.zeros(0, np.int32)
    return np.stack(desc), np.asarray(xy), np.asarray(tt, np.int32)


def color_of(frames):
    desc, xy, tt = [], [], []
    for t, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        for r in range(4):
            for c in range(4):
                cell = hsv[r * h // 4:(r + 1) * h // 4, c * w // 4:(c + 1) * w // 4]
                cell = cell.reshape(-1, 3).astype(np.float64)
                if len(cell) == 0:
                    continue
                hue = np.minimum((cell[:, 0] / 180.0 * 12).astype(int), 11)
                sat = np.minimum((cell[:, 1] / 256.0 * 4).astype(int), 3)
                val = cell[:, 2] / 255.0
                dark = val < 0.15
                idx = np.where(dark, 48 + np.minimum((val * 4 / 0.15).astype(int), 3), hue * 4 + sat)
                hist = np.bincount(idx, minlength=52).astype(np.float64)
                desc.append(hist / hist.sum())
                xy.append(((c + 0.5) / 4, (r + 0.5) / 4))
                tt.append(t)
    return np.stack(desc), np.asarray(xy), np.asarray(tt, np.int32)


def assign(x, cents):
    if len(x) == 0:
        return np.zeros(0, np.int32)
    d = (x * x).sum(1)[:, None] - 2 * x @ cents.T + (cents * cents).sum(1)[None, :]
    return d.argmin(1).astype(np.int32)


def kmeans(x, k, seed, iters=20):
    rng = np.random.default_rng(seed)
    fit = x[rng.choice(len(x), min(len(x), 6000), replace=False)]
    k = min(k, len(fit))
    cents = fit[rng.choice(len(fit), k, replace=False)].copy()
    for _ in range(iters):
        lab = assign(fit, cents)
        for j in range(k):
            take = lab == j
            cents[j] = fit[take].mean(0) if np.any(take) else fit[rng.integers(len(fit))]
    return cents


def volumes(xy, tt, nframes, words, vocab):
    level0 = np.full((1, vocab), 1e-3)
    level1 = np.full((8, vocab), 1e-3)
    if len(words):
        np.add.at(level0, (np.zeros(len(words), int), words), 1.0)
        tb = np.minimum((tt / max(nframes - 1, 1) * 2).astype(int), 1)
        sx = np.minimum((xy[:, 0] * 2).astype(int), 1)
        sy = np.minimum((xy[:, 1] * 2).astype(int), 1)
        np.add.at(level1, (tb * 4 + sy * 2 + sx, words), 1.0)
    level0 /= level0.sum(1, keepdims=True)
    level1 /= level1.sum(1, keepdims=True)
    return level0, level1


def chi2(a, b):
    diff = a[:, None, :] - b[None, :, :]
    return 0.5 * np.sum(diff * diff / (a[:, None, :] + b[None, :, :] + 1e-10), -1)


def level0_distance(hist):
    x = hist[:, 0, :]
    return chi2(x, x)


def level1_distance(hist):
    n = len(hist)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cost = chi2(hist[i], hist[j])
            row, col = linear_sum_assignment(cost)
            dist[i, j] = dist[j, i] = cost[row, col].mean()
    return dist


def extract(root, cache):
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    items = list_clips(root)
    feats = []
    for i, (_, _, path) in enumerate(items):
        frames = sample_frames(path)
        feats.append((sift_of(frames), hof_of(frames), color_of(frames), len(frames)))
        if i % 20 == 0:
            print(f"frames {i + 1}/{len(items)}", flush=True)
    src = [i for i, (domain, _, _) in enumerate(items) if domain == "source"]
    books = {}
    for fi, (name, k, seed) in enumerate((("sift", SIFT_K, 0), ("hof", HOF_K, 1), ("color", COLOR_K, 2))):
        blocks = [feats[i][fi][0] for i in src if len(feats[i][fi][0])]
        books[name] = kmeans(np.concatenate(blocks), k, seed)
    out = {}
    for fi, name in enumerate(("sift", "hof", "color")):
        vocab = len(books[name])
        level0, level1 = [], []
        for feat in feats:
            desc, xy, tt = feat[fi]
            words = assign(desc, books[name])
            h0, h1 = volumes(xy, tt, feat[3], words, vocab)
            level0.append(h0)
            level1.append(h1)
        out[f"{name}__L0"] = level0_distance(np.stack(level0))
        out[f"{name}__L1"] = level1_distance(np.stack(level1))
    out["labels"] = np.array([item[1] for item in items])
    out["domains"] = np.array([item[0] for item in items])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out
