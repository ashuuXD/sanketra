import json
from pathlib import Path
import cv2
import numpy as np
from training.prepare_dataset import prepare
from evaluation.robustness import build_corrupted, run


def make_src(tmp: Path, n_groups=10, per=4):
    (tmp / "images").mkdir(parents=True); (tmp / "labels").mkdir()
    rng = np.random.default_rng(0)
    for g in range(n_groups):
        base = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        for k in range(per):
            name = f"seq{g}_{k}"
            cv2.imwrite(str(tmp / "images" / f"{name}.png"), np.clip(base.astype(int) + k * 3, 0, 255).astype(np.uint8))
            (tmp / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.3 0.3")
    cv2.imwrite(str(tmp / "images" / "seqX_0.png"), np.zeros((64, 64, 3), np.uint8))
    (tmp / "images" / "broken.png").write_bytes(b"not an image")
    (tmp / "labels" / "seqX_0.txt").write_text("9 2 2 2 2")      # invalid class + coords


def test_prepare_group_split_no_leakage_and_quality(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_src(src)
    s = prepare(src, dst, ["sign"], r"^(seq\d+)_", seed=1)
    assert s["quality"]["n_corrupt"] == 1 and s["quality"]["n_bad_labels"] == 1
    where = {}
    for split in ("train", "val", "test"):
        for p in (dst / "images" / split).glob("*.png"):
            where.setdefault(p.stem.split("_")[0], set()).add(split)
    assert all(len(v) == 1 for v in where.values())        # each sequence lives in exactly one split
    assert sum(v["images"] for v in s["splits"].values()) == 40
    assert (dst / "data.yaml").exists() and json.loads((dst / "dataset_stats.json").read_text())["seed"] == 1


def test_robustness_pipeline_with_stub_evaluator(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_src(src); prepare(src, dst, ["sign"], r"^(seq\d+)_", seed=1)
    import evaluation.robustness as rb
    monkeypatch.setattr(rb, "RESULTS", tmp_path / "res"); monkeypatch.setattr(rb, "TMP", tmp_path / "tmp")
    calls = []
    out = run("stub", dst / "data.yaml", "test", ["blur", "noise"], [0, 3], 640, None, "t",
              evaluate=lambda y: calls.append(y) or {"precision": 1, "recall": 1, "map50": 0.5, "map50_95": 0.3})
    assert len(calls) == 4 and (tmp_path / "res" / "robustness_t.json").exists() and (tmp_path / "res" / "robustness_t.png").exists()
    assert set(out["results"]) == {"blur", "noise"}


def test_gtsdb_adapter_category_and_full(tmp_path):
    import cv2, numpy as np
    from training.adapters.gtsdb import convert
    src = tmp_path / "g"; src.mkdir()
    for i in range(4):
        cv2.imwrite(str(src / f"{i:05d}.ppm"), np.full((80, 136, 3), 90 + i, np.uint8))
    (src / "gt.txt").write_text("00000.ppm;10;10;30;30;4\n00000.ppm;50;20;70;40;11\n00001.ppm;5;5;25;25;33\n00002.ppm;60;10;90;40;6\n")
    r = convert(src, tmp_path / "o1", "category")
    assert r["images"] == 4 and r["boxes"] == 4 and r["empty_images"] == 1 and r["classes"] == 4
    first = (tmp_path / "o1" / "labels" / "blk000_00000.txt").read_text().splitlines()
    assert first[0].split()[0] == "0" and first[1].split()[0] == "1"          # 4 speed limit -> prohibitory, 11 -> danger
    assert (tmp_path / "o1" / "labels" / "blk000_00001.txt").read_text().split()[0] == "2"   # 33 -> mandatory
    assert (tmp_path / "o1" / "labels" / "blk000_00002.txt").read_text().split()[0] == "3"   # 6 -> other
    r2 = convert(src, tmp_path / "o2", "full", include_empty=False)
    assert r2["images"] == 3 and r2["classes"] == 43


def test_temporal_states_ema_and_hold_reduce_flicker():
    from evaluation.temporal_eval import track_states
    from vision.temporal import flicker_rate
    # confidence wobbles around the 0.5 threshold, with one missed frame
    obs = [(0, .6), (1, .4), (2, .55), (3, .45), (4, .6), (6, .58), (7, .42), (8, .6)]
    raw, ema, held = track_states(obs, 0.5, 0.5, 3)
    assert flicker_rate(ema) < flicker_rate(raw)
    assert flicker_rate(held) <= flicker_rate(ema)
    assert len(raw) == 9 and held[5] is True            # the missing frame 5 is bridged by the hold


def _img(path, w=200, h=120, v=100):
    import cv2, numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), v, np.uint8))


def test_indian_import_normalises_names():
    from training.adapters.indian_import import norm_name
    assert norm_name("1. Gap in Median") == "gap_in_median" and norm_name("  12_No Entry!! ") == "no_entry"


def test_indian_import_yolo_layout_and_voc_layout_and_folders(tmp_path):
    from training.adapters.indian_import import convert, detect_layout
    y = tmp_path / "y"; _img(y / "a.png"); (y / "a.txt").write_text("1 0.5 0.5 0.2 0.3"); (y / "classes.txt").write_text("Stop\n2. Give Way\n")
    assert detect_layout(y) == "yolo"
    r = convert(y, tmp_path / "oy"); assert r["images_written"] == 1 and r["per_class"] == {"give_way": 1}
    v = tmp_path / "v"; _img(v / "b.jpg")
    (v / "b.xml").write_text("<annotation><object><name>No Entry</name><bndbox><xmin>20</xmin><ymin>10</ymin><xmax>100</xmax><ymax>60</ymax></bndbox></object></annotation>")
    assert detect_layout(v) == "voc"
    r = convert(v, tmp_path / "ov"); assert r["per_class"] == {"no_entry": 1} and (tmp_path / "ov" / "names.txt").read_text() == "no_entry"
    f = tmp_path / "f"; _img(f / "1. Gap in Median" / "1.png"); _img(f / "2. Speed Limit" / "2.png"); _img(f / "2. Speed Limit" / "3.png")
    assert detect_layout(f) == "folders"
    import pytest
    with pytest.raises(SystemExit):
        convert(f, tmp_path / "of0")
    calls = []

    def propose(img):
        calls.append(1); return (20, 10, 100, 60, 0.9) if len(calls) != 3 else None      # third image: nothing found
    r = convert(f, tmp_path / "of", propose)
    assert r["machine_made_labels"] is True and r["images_written"] == 2 and r["skipped"] == 1 and set(r["per_class"]) == {"gap_in_median", "speed_limit"}


def test_registry_sync_adds_unverified_stubs_in_home(tmp_path, monkeypatch):
    import json
    import training.registry_sync as rs
    monkeypatch.setattr(rs, "HOME", tmp_path)
    assert rs.sync(["stop sign", "gap_in_median", "gap_in_median"], "test") == 1          # existing and duplicate names are skipped
    e = json.loads((tmp_path / "sign_registry.local.json").read_text())["signs"][0]
    assert e["class_name"] == "gap_in_median" and e["verified"] is False and e["needs_review"] is True


def test_indian_import_numeric_folder_names_stay_distinct_and_tiny_crops_are_refused(tmp_path):
    import cv2, numpy as np, pytest
    from training.adapters.indian_import import norm_name, convert
    assert norm_name("7") == "class_7" and norm_name("12") == "class_12" and norm_name("7") != norm_name("8")
    for c in ("0", "1"):
        d = tmp_path / "src" / c; d.mkdir(parents=True)
        for k in range(3):
            cv2.imwrite(str(d / f"{k}.png"), np.full((32, 32, 3), 120 + k, np.uint8))
    with pytest.raises(SystemExit) as e:
        convert(tmp_path / "src", tmp_path / "dst", propose=lambda img: (0, 0, 31, 31, 0.9), layout="folders")
    assert "smaller than 64 px" in str(e.value)


def test_gtsdb_category_checker_parses_readme_and_flags_mismatch():
    import json
    from training.adapters.check_gtsdb_categories import parse_readme, compare, CFG
    r = parse_readme("0 = speed limit 20 \n(prohibitory)\n6 = end of speed limit 80 (other)\n10 = no overtaking (trucks)\n(prohibitory)\n"
                     "33 = go right (mandatory)\n41 = restriction ends (overtaking)\n(other)\n42 = restriction ends (overtaking (trucks))\n(other)\n")
    assert r[0][1] == "prohibitory" and r[6][1] == "other" and r[33][1] == "mandatory"
    assert r[10][1] == "prohibitory" and r[41][1] == "other" and r[42][1] == "other"   # brackets inside names must not be read as categories
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    full = {i: ("n", "other") for i in range(43)}
    for cat in cfg["names"][:-1]:
        for c in cfg[cat]:
            full[c] = ("n", cat)
    assert compare(full, cfg) == []
    full[6] = ("end", "prohibitory")
    assert len(compare(full, cfg)) == 1


def test_seed_summary_mean_sd_and_interval():
    from evaluation.seed_summary import summarise
    s = summarise([0.90, 0.92, 0.94])
    assert s["n"] == 3 and abs(s["mean"] - 0.92) < 1e-9 and abs(s["sd"] - 0.02) < 1e-9
    assert s["ci95"][0] < 0.92 < s["ci95"][1] and abs(s["ci95"][1] - (0.92 + 4.303 * 0.02 / 3 ** 0.5)) < 1e-9   # t(0.975, df=2) = 4.303
    assert "sd" not in summarise([0.9])                      # one run: no spread can be estimated
