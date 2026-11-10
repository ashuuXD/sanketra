"""Import an Indian road-sign dataset (whatever layout it arrives in) into YOLO format.

Detected layouts, in this order:
  yolo     images with a same-name .txt label (beside the image or in a parallel labels/ folder);
           class names from classes.txt / names.txt / data.yaml if present, else class_<n>
  voc      images with a same-name Pascal VOC .xml (class names from the <name> tags)
  folders  one folder per class and NO boxes (e.g. "1. Gap in Median/1.png").
           Boxes are PROPOSED by a detector (--proposal-model): the best box in each image gets the folder's class.
           This assumes one main sign per image. These are machine-made labels: review them (see import_report.json)
           and say so in your report.
Run: python -m training.adapters.indian_import --src <folder> --dst <SanketraData>\\datasets\\raw\\indian [--proposal-model best.pt]
Check the dataset's licence and citation requirements before using or sharing it.
"""
from __future__ import annotations
import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Callable
import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NON_LABEL = {"classes", "names", "readme", "labels", "obj", "train", "val", "test"}


def norm_name(s: str) -> str:
    """'1. Gap in Median' -> 'gap_in_median' (the same function is used for registry entries so names match the model)."""
    raw = s.strip()
    s = re.sub(r"^[\s\d\.\-_)]+", "", raw)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    if not s:                                  # a folder called just "7": keep the number, never collapse all of them into one class
        digits = re.sub(r"\D", "", raw)
        return f"class_{digits}" if digits else "unnamed"
    return s


def _images(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)


def _find_label(img: Path, ext: str, src: Path) -> Path | None:
    for cand in (img.with_suffix(ext), img.parent.parent / "labels" / (img.stem + ext), src / "labels" / (img.stem + ext)):
        if cand.exists() and cand.stem.lower() not in NON_LABEL:
            return cand
    return None


def _names_from_files(src: Path) -> list[str] | None:
    for n in ("classes.txt", "names.txt", "obj.names"):
        for p in src.rglob(n):
            return [norm_name(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    for p in src.rglob("data.yaml"):
        import yaml
        names = (yaml.safe_load(p.read_text(encoding="utf-8-sig")) or {}).get("names")
        if isinstance(names, dict):
            return [norm_name(str(names[k])) for k in sorted(names)]
        if isinstance(names, list):
            return [norm_name(str(x)) for x in names]
    return None


def detect_layout(src: Path) -> str:
    imgs = _images(src)
    if not imgs:
        raise SystemExit(f"no images found under {src}")
    if any(_find_label(i, ".txt", src) for i in imgs[:200]):
        return "yolo"
    if any(_find_label(i, ".xml", src) for i in imgs[:200]):
        return "voc"
    if len({i.parent for i in imgs}) > 1:
        return "folders"
    raise SystemExit("images found but no labels and no class folders: nothing to learn from. See docs/INDIAN_SIGNS_PLAN.md to label them.")


def _save(img, out: Path, stem: str, lines: list[str], max_side: int) -> None:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w); img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "labels" / f"{stem}.txt").write_text("\n".join(lines))


def convert(src: Path, dst: Path, propose: Callable | None = None, max_side: int = 1280, layout: str | None = None) -> dict:
    layout = layout or detect_layout(src)
    (dst / "images").mkdir(parents=True, exist_ok=True); (dst / "labels").mkdir(parents=True, exist_ok=True)
    skipped, per_class, n = [], Counter(), 0
    imgs = _images(src)
    if layout == "yolo":
        names = _names_from_files(src)
        for i, p in enumerate(imgs):
            lab = _find_label(p, ".txt", src); img = cv2.imread(str(p))
            if lab is None or img is None:
                skipped.append(f"{p.name}: no label or unreadable"); continue
            rows = [l.split() for l in lab.read_text().splitlines() if len(l.split()) >= 5]
            if not rows:
                skipped.append(f"{p.name}: empty label"); continue
            if names is None:
                names = [f"class_{k}" for k in range(1 + max(int(float(r[0])) for r in rows))]
            lines = []
            for r in rows:
                c = int(float(r[0]))
                while c >= len(names):
                    names.append(f"class_{len(names)}")
                lines.append(" ".join([str(c)] + r[1:5])); per_class[names[c]] += 1
            _save(img, dst, f"img_{i:05d}", lines, max_side); n += 1
    elif layout == "voc":
        seen: dict[str, int] = {}
        for i, p in enumerate(imgs):
            lab = _find_label(p, ".xml", src); img = cv2.imread(str(p))
            if lab is None or img is None:
                skipped.append(f"{p.name}: no xml or unreadable"); continue
            h, w = img.shape[:2]; lines = []
            for o in ET.parse(lab).getroot().iter("object"):
                nm = norm_name(o.findtext("name", "")); b = o.find("bndbox")
                if b is None:
                    continue
                x1, y1, x2, y2 = (float(b.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
                c = seen.setdefault(nm, len(seen))
                lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"); per_class[nm] += 1
            if not lines:
                skipped.append(f"{p.name}: no objects"); continue
            _save(img, dst, f"img_{i:05d}", lines, max_side); n += 1
        names = list(seen)
    else:  # folders
        tiny = sum(1 for p in imgs[::max(len(imgs) // 200, 1)] if (im := cv2.imread(str(p))) is not None and min(im.shape[:2]) < 64)
        sampled = len(imgs[::max(len(imgs) // 200, 1)])
        if sampled and tiny / sampled > 0.5:
            raise SystemExit("most images are smaller than 64 px: this is a cropped-sign CLASSIFICATION set, not road scenes. "
                             "A detector cannot learn from it (boxes would just be the whole image). See docs/INDIAN_SIGNS_PLAN.md, section 'Kaggle Indian Traffic Sign Dataset'.")
        if propose is None:
            raise SystemExit("this dataset has class folders but no boxes. Pass --proposal-model <weights> to let a detector propose boxes (then review them).")
        names = sorted({norm_name(p.parent.name) for p in imgs})
        for i, p in enumerate(imgs):
            img = cv2.imread(str(p)); cls = norm_name(p.parent.name)
            if img is None:
                skipped.append(f"{p.name}: unreadable"); continue
            box = propose(img)                                    # (x1, y1, x2, y2, conf) or None
            if box is None:
                skipped.append(f"{p.parent.name}/{p.name}: no sign proposed"); continue
            h, w = img.shape[:2]; x1, y1, x2, y2, conf = box
            line = f"{names.index(cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            _save(img, dst, f"{cls}_{i:05d}", [line], max_side); per_class[cls] += 1; n += 1
    (dst / "names.txt").write_text("\n".join(names))
    report = {"layout": layout, "images_written": n, "skipped": len(skipped), "classes": len(names), "per_class": dict(per_class),
              "machine_made_labels": layout == "folders", "skipped_examples": skipped[:50]}
    (dst / "import_report.json").write_text(json.dumps(report, indent=2))
    return report


def detector_proposer(weights: str, conf: float = 0.05, device=None):
    """Class-agnostic: the most confident box of any class in the image."""
    from ultralytics import YOLO
    m = YOLO(weights)

    def propose(img):
        r = m.predict(img, conf=conf, verbose=False, device=device)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax()); x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
        return x1, y1, x2, y2, float(r.boxes.conf[i])
    return propose


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True); ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--proposal-model", default=None, help="weights used to propose boxes when the dataset has class folders but no boxes")
    ap.add_argument("--max-side", type=int, default=1280); ap.add_argument("--layout", choices=["yolo", "voc", "folders"], default=None)
    a = ap.parse_args()
    lay = a.layout or detect_layout(a.src); print("detected layout:", lay)
    r = convert(a.src, a.dst, detector_proposer(a.proposal_model) if (lay == "folders" and a.proposal_model) else None, a.max_side, lay)
    print(json.dumps({k: r[k] for k in ("layout", "images_written", "skipped", "classes")}, indent=2))
    if r["machine_made_labels"]:
        print("NOTE: boxes were proposed by a model. Review a sample before training and say so in your report.")
    print("next: python -m training.prepare_dataset --src", a.dst, "--dst <prepared> --names", a.dst / "names.txt")
