import os
import re
import shutil
import tarfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

CLASSES = ("animal", "water", "city", "vehicle")
QUOTA = {"source": 30, "target": 24}
PER_KEYWORD = 3
DUP_RATIO = 0.66
CAPTION_WINDOW = 140
VEHICLE_WINDOW = 80

ANIMALS = (
    "alligator", "antelope", "baboon", "bear", "bird", "bison", "buffalo", "butterfly",
    "camel", "cat", "cheetah", "cow", "crane", "crocodile", "deer", "dog", "dolphin",
    "duck", "eagle", "elephant", "flamingo", "fox", "frog", "giraffe", "goat", "goose",
    "gorilla", "heron", "hippo", "hippopotamus", "horse", "hyena", "iguana", "jellyfish",
    "kangaroo", "koala", "leopard", "lion", "macaque", "monkey", "moose", "octopus",
    "orangutan", "otter", "owl", "panda", "parrot", "peacock", "penguin", "rabbit",
    "raccoon", "rhino", "rhinoceros", "seagull", "seal", "shark", "sheep", "skunk",
    "snake", "spider", "squirrel", "swan", "tiger", "turtle", "whale", "wolf", "zebra",
)
WATER = (
    "beach", "coast", "coastline", "lake", "ocean", "pond", "reef", "river", "sea",
    "stream", "waterfall",
)
CITIES = ("cityscape", "downtown", "metropolis", "skyscraper", "skyline")
VEHICLES = (
    "aircraft", "airplane", "bicycle", "boat", "bus", "car", "helicopter", "locomotive",
    "motorcycle", "plane", "sedan", "ship", "taxi", "train", "truck", "vehicle",
)

PERSON_START = re.compile(
    r"^(?:a |an |the )?(?:young |old )?(?:person|man|woman|people|boy|girl|child|someone|he|she)\b",
    re.I,
)
ANIMAL_REJECT = (
    re.compile(r"\bmt\.?\s+buffalo\b", re.I),
    re.compile(r"\bbutterfly fish\b", re.I),
    re.compile(r"\bconstruction crane\b", re.I),
    re.compile(r"\bsnakes along\b", re.I),
    re.compile(r"\b(?:statue|statues|carousel|merry-go-round|stuffed animal)\b", re.I),
)
WAVE = re.compile(r"\bwaves?\b", re.I)
WAVE_VERB = re.compile(r"\b(?:he|she|they|person|man|woman|boy|girl)\s+waves\b", re.I)


def _word(word, stem=False):
    if word.endswith("y") and stem:
        return re.compile(r"\b" + re.escape(word[:-1]) + r"(?:y|ies)\b", re.I)
    if word.endswith(("s", "x", "ch", "sh")):
        return re.compile(r"\b" + re.escape(word) + r"(?:es)?\b", re.I)
    if stem:
        return re.compile(r"\b" + re.escape(word) + r"\w*\b", re.I)
    return re.compile(r"\b" + re.escape(word) + r"s?\b", re.I)


ANIMAL_RE = [(word, _word(word, stem=True)) for word in ANIMALS]
WATER_RE = [(word, _word(word)) for word in WATER]
CITY_RE = [(word, _word(word)) for word in CITIES]
VEHICLE_RE = [(word, _word(word)) for word in VEHICLES]


def earliest(text, table):
    found = None
    at = None
    for word, cre in table:
        match = cre.search(text)
        if match and (at is None or match.start() < at):
            found, at = word, match.start()
    return found


def classify(caption):
    text = " ".join(caption.split())
    window = text[:CAPTION_WINDOW]
    animal = earliest(window, ANIMAL_RE)
    if animal is not None:
        if any(cre.search(text) for cre in ANIMAL_REJECT):
            return None
        if PERSON_START.search(text):
            return None
        return "animal", animal
    if earliest(window, WATER_RE) is not None or (WAVE.search(window) and not WAVE_VERB.search(window)):
        if PERSON_START.search(text):
            return None
        keyword = earliest(window, WATER_RE) or "wave"
        return "water", keyword
    city = earliest(window, CITY_RE)
    if city is not None:
        return "city", city
    vehicle = earliest(text[:VEHICLE_WINDOW], VEHICLE_RE)
    if vehicle is not None:
        return "vehicle", vehicle
    return None


def near_duplicate(a, b):
    def head(text):
        sentence = re.split(r"[.!?]", text.strip(), maxsplit=1)[0]
        return " ".join(sentence.lower().split())[:240]
    return SequenceMatcher(None, head(a), head(b)).ratio() >= DUP_RATIO


def _pairs_from_names(names):
    grouped = {}
    for name in names:
        base = Path(name).name
        if not base or base.startswith("."):
            continue
        suffix = Path(base).suffix.lower()
        if suffix not in (".mp4", ".txt"):
            continue
        grouped.setdefault(Path(base).stem, {})[suffix] = name
    rows = []
    for stem in sorted(grouped):
        item = grouped[stem]
        if ".mp4" in item and ".txt" in item:
            rows.append((stem, item[".txt"], item[".mp4"]))
    return rows


def iter_shards(domain_root):
    domain_root = Path(domain_root)
    tars = sorted(p for p in domain_root.glob("*.tar") if p.is_file())
    if tars:
        for tar_path in tars:
            yield tar_path.stem, ("tar", tar_path)
        return
    nested = sorted(p for p in domain_root.iterdir() if p.is_dir() and any(p.glob("*.mp4")))
    if nested:
        for folder in nested:
            yield folder.name, ("dir", folder)
        return
    for mp4 in sorted(domain_root.glob("*.mp4")):
        yield mp4.stem, ("file", mp4)


def read_candidates(kind, path):
    if kind == "tar":
        with tarfile.open(path, "r:") as handle:
            names = [member.name for member in handle.getmembers() if member.isfile()]
            rows = []
            for stem, txt_name, mp4_name in _pairs_from_names(names):
                extracted = handle.extractfile(txt_name)
                if extracted is None:
                    continue
                caption = extracted.read().decode("utf-8", "replace")
                rows.append((stem, caption, mp4_name))
        return rows
    if kind == "dir":
        rows = []
        for stem, txt_name, mp4_name in _pairs_from_names(p.name for p in Path(path).iterdir()):
            caption = (Path(path) / txt_name).read_text(encoding="utf-8", errors="replace")
            rows.append((stem, caption, mp4_name))
        return rows
    txt = Path(path).with_suffix(".txt")
    if not txt.is_file():
        return []
    caption = txt.read_text(encoding="utf-8", errors="replace")
    return [(Path(path).stem, caption, Path(path).name)]


def copy_video(kind, path, mp4_name, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if kind == "tar":
        with tarfile.open(path, "r:") as handle:
            extracted = handle.extractfile(mp4_name)
            if extracted is None:
                raise RuntimeError(f"missing video {Path(mp4_name).name}")
            with dest.open("wb") as out:
                shutil.copyfileobj(extracted, out)
        return
    src = path if kind == "file" else Path(path) / mp4_name
    shutil.copyfile(src, dest)


def select_domain(domain_root, quota):
    chosen = {label: [] for label in CLASSES}
    kept_text = {label: [] for label in CLASSES}
    used_shards = set()
    keyword_count = {label: Counter() for label in CLASSES}
    for shard, (kind, path) in iter_shards(domain_root):
        if shard in used_shards:
            continue
        if all(len(chosen[label]) >= quota for label in CLASSES):
            break
        picked = None
        for stem, caption, mp4_name in read_candidates(kind, path):
            labeled = classify(caption)
            if labeled is None:
                continue
            label, keyword = labeled
            if len(chosen[label]) >= quota:
                continue
            if label in ("animal", "vehicle") and keyword_count[label][keyword] >= PER_KEYWORD:
                continue
            if any(near_duplicate(caption, prev) for prev in kept_text[label]):
                continue
            picked = (label, keyword, stem, caption, kind, path, mp4_name)
            break
        if picked is None:
            continue
        label, keyword, stem, caption, kind, path, mp4_name = picked
        chosen[label].append((stem, kind, path, mp4_name))
        kept_text[label].append(caption)
        keyword_count[label][keyword] += 1
        used_shards.add(shard)
    for label in CLASSES:
        if len(chosen[label]) < quota:
            raise RuntimeError(f"need {quota} {label} clips, found {len(chosen[label])}")
    return chosen


def prepare(raw_root, out_root):
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    for domain, quota in QUOTA.items():
        domain_root = raw_root / domain
        if not domain_root.is_dir():
            raise RuntimeError(f"missing {domain} folder")
        chosen = select_domain(domain_root, quota)
        for label in CLASSES:
            dest_dir = out_root / domain / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for old in dest_dir.glob("*.mp4"):
                old.unlink()
            for stem, kind, path, mp4_name in chosen[label]:
                copy_video(kind, path, mp4_name, dest_dir / f"{stem}.mp4")
        n = sum(len(chosen[label]) for label in CLASSES)
        print(f"{domain} {n}", flush=True)
    return out_root


def main():
    raw = os.environ.get("RAW_ROOT")
    if not raw:
        raise SystemExit("set RAW_ROOT")
    prepare(raw, os.environ.get("DATA_ROOT", "data"))


if __name__ == "__main__":
    main()
