"""Compare configs/gtsdb_categories.json with the category written next to each class in GTSDB's own ReadMe.txt.
Usage:  python -m training.adapters.check_gtsdb_categories --src "C:\\path\\FullIJCNN2013"
Prints every class whose category differs, and exits with code 1 if any differ (nothing is changed automatically)."""
import argparse, json, re, sys
from pathlib import Path

CFG = Path(__file__).resolve().parents[2] / "configs" / "gtsdb_categories.json"


CATEGORIES = {"prohibitory", "danger", "mandatory", "other"}


def parse_readme(text: str) -> dict[int, tuple[str, str]]:
    """{class_id: (name, category)}. An entry starts at 'N =' and runs to the next entry. Names may contain brackets
    ('no overtaking (trucks)', even nested), so the category is the LAST bracketed word that is a known category."""
    starts = list(re.finditer(r"(?m)^\s*(\d{1,2})\s*=", text))
    out = {}
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else min(len(text), m.end() + 160)
        body = text[m.end():end]
        cats = [c.group(1).lower() for c in re.finditer(r"\(\s*([A-Za-z]+)\s*\)", body) if c.group(1).lower() in CATEGORIES]
        if cats:
            name = re.split(r"\(\s*(?:%s)\s*\)" % "|".join(CATEGORIES), body, flags=re.I)[0].strip()
            out[int(m.group(1))] = (" ".join(name.split()), cats[-1])
    return out


def config_category(cfg: dict, cid: int) -> str:
    for cat in cfg["names"][:-1]:
        if cid in cfg[cat]:
            return cat
    return cfg["names"][-1]            # everything not listed is "other"


def compare(readme: dict, cfg: dict) -> list[str]:
    problems = []
    for cid in range(43):
        if cid not in readme:
            problems.append(f"class {cid}: not found in ReadMe (parser could not read it)")
            continue
        name, cat = readme[cid]
        mine = config_category(cfg, cid)
        if cat != mine:
            problems.append(f"class {cid} ({name}): ReadMe says '{cat}', configs/gtsdb_categories.json says '{mine}'")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True)
    a = ap.parse_args()
    rd = next(iter(Path(a.src).rglob("ReadMe*.txt")), None)
    if rd is None:
        sys.exit(f"ReadMe.txt not found under {a.src}")
    readme = parse_readme(rd.read_text(errors="ignore"))
    print(f"parsed {len(readme)} classes from {rd}")
    probs = compare(readme, json.loads(CFG.read_text(encoding="utf-8")))
    print("\n".join(probs) if probs else "OK: all 43 classes match the ReadMe categories")
    sys.exit(1 if probs else 0)
