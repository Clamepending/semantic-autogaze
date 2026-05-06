"""Aggregate the phase18 hard-neg-mining sweep results.

After scripts/eval_phase18_sweep.sh runs, this script reads all
`results/eval_phase18/*/coco50.json` and the corresponding 8-image
grid mIoU from `results/eval_phase18/*/sweep.log` (re-parsing the
'>>> mIoU = ...' lines indexed by '[sweep] eval ...' markers).

Then it ranks all variants by:
  - 8-image qual-grid mIoU (canonical Phase 15 metric)
  - 50-image follow-up mIoU
  - Per-category lift on FAILURE_CATEGORIES (knife/skis/tie/baseball-bat/
    snowboard/sports-ball/skateboard/spoon)

Compares against the v0.5.0 reference (phase15 atto step 35000):
  8-img = 0.722, 50-img = 0.681

Usage:
  /home/ogata/miniconda3/envs/hunter/bin/python -m scripts.aggregate_phase18
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path("/home/ogata/semantic-autogaze/results/eval_phase18")
V050_REF = {
    "8img_miou": 0.722,
    "50img": json.load(open("/home/ogata/semantic-autogaze/results/eval_phase15/atto_long_50k_step35000/coco50.json")),
}

FAILURE_CATEGORIES = [
    "knife", "skis", "tie", "baseball bat",
    "snowboard", "sports ball", "skateboard", "spoon",
]


def main():
    sweep_log = (ROOT / "sweep.log").read_text() if (ROOT / "sweep.log").exists() else ""
    # 8-image mIoU values are printed by eval_phase2_ckpt as "  >>> mIoU = ..."
    # interleaved with [sweep] eval <slug> markers that appear in stdout (saved
    # in sweep.log as well via tee/redirect).
    starts = re.findall(r"\[sweep\][^\n]+ eval (\S+)", sweep_log)
    miou_lines = re.findall(r">>> mIoU = ([\d.]+)", sweep_log)
    eight_img = dict(zip(starts, miou_lines))

    rows = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        slug = d.name
        coco50_p = d / "coco50.json"
        if not coco50_p.exists():
            continue
        c = json.load(open(coco50_p))
        miou_8 = float(eight_img.get(slug, "nan"))
        miou_50 = float(c["miou"])
        per_cat = c["per_cat"]
        # Lift on failure cats (50-img, mean over the 8 categories that are present)
        ref_per_cat = V050_REF["50img"]["per_cat"]
        lifts = []
        for fc in FAILURE_CATEGORIES:
            if fc in per_cat and fc in ref_per_cat:
                lifts.append(per_cat[fc] - ref_per_cat[fc])
        fail_lift_mean = sum(lifts) / max(len(lifts), 1)
        rows.append((slug, miou_8, miou_50, fail_lift_mean, per_cat))

    print(f"\n{'slug':40s} {'8-img':>6s} Δ8     {'50-img':>7s} Δ50    {'fail-lift':>10s}")
    print("-" * 90)
    print(f"{'v0.5.0 (phase15 step 35000)':40s} {V050_REF['8img_miou']:>6.3f} {0:+.3f} "
          f"{V050_REF['50img']['miou']:>7.3f} {0:+.3f} {0:+10.3f}")
    rows.sort(key=lambda r: -r[3])
    for slug, m8, m50, lift, _ in rows:
        d8 = m8 - V050_REF["8img_miou"]
        d50 = m50 - V050_REF["50img"]["miou"]
        print(f"{slug:40s} {m8:>6.3f} {d8:+.3f} {m50:>7.3f} {d50:+.3f} {lift:+10.3f}")

    # Per-category breakdown for the top-ranked variant
    if rows:
        top = rows[0]
        print(f"\n=== Per-category 50-image lift for top variant: {top[0]} ===")
        ref_pc = V050_REF["50img"]["per_cat"]
        top_pc = top[4]
        rowsd = []
        for cat in sorted(set(top_pc) & set(ref_pc)):
            rowsd.append((cat, ref_pc[cat], top_pc[cat], top_pc[cat] - ref_pc[cat]))
        rowsd.sort(key=lambda r: r[3])  # worst first
        print(f"{'category':28s} {'v0.5.0':>8s} {'top':>8s} {'Δ':>7s}")
        for cat, a, b, d in rowsd:
            mark = " <<< failure cat" if cat in FAILURE_CATEGORIES else ""
            print(f"{cat:28s} {a:>8.3f} {b:>8.3f} {d:+7.3f}{mark}")


if __name__ == "__main__":
    main()
