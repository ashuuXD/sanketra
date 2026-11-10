"""Summarise several training runs that differ only in the training seed.
Usage: python -m evaluation.seed_summary --tags gtsdb_n100 gtsdb_n100_s1 gtsdb_n100_s2
Reads experiments/results/detection_<tag>.json (test split) and prints mean, sample standard deviation, min and max of
mAP@0.5 and mAP@0.5:0.95, plus a 95% t-interval of the mean. With 3 seeds the interval is wide: that is the honest result."""
import argparse, json, statistics as st
from pathlib import Path



def summarise(values: list[float]) -> dict:
    n = len(values)
    out = {"n": n, "mean": st.fmean(values), "min": min(values), "max": max(values)}
    if n >= 2:
        sd = st.stdev(values)
        df = n - 1
        t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}.get(df, 1.96)
        half = t * sd / n ** 0.5
        out.update({"sd": sd, "ci95": [out["mean"] - half, out["mean"] + half]})
    return out


def load(tag: str, results: Path) -> dict:
    return json.loads((results / f"detection_{tag}.json").read_text())["overall"]


if __name__ == "__main__":
    from backend.settings import HOME                      # imported here so summarise() can be tested without the web stack
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--results", default=str(HOME / "experiments" / "results"))
    a = ap.parse_args()
    runs = [load(t, Path(a.results)) for t in a.tags]
    for key in ("map50", "map50_95"):
        s = summarise([r[key] for r in runs])
        line = f"{key:9} n={s['n']}  mean {s['mean']:.3f}  min {s['min']:.3f}  max {s['max']:.3f}"
        if "sd" in s:
            line += f"  sd {s['sd']:.3f}  95% CI [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"
        print(line)
    print("Seeds change training only; the test split is fixed (120 images, 160 boxes). The interval does not cover test-set sampling error.")
