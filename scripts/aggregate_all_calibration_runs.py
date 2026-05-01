"""Aggregate phase18 + phase19 evaluation results into one ranked table.

Reads coco50.json from every results/eval_phase18/* and results/eval_phase19/*
sub-directory, ranks against the v0.5.0 phase15 step-35000 baseline, and
prints a single table that picks the best variant by composite score:
   composite = 50-img mIoU + (failure-cat-mean-lift) - max(0, -strong-cat-lift) * 2

The composite leans toward variants that lift failure cats without sacrificing
strong cats. A tie on overall mIoU + a +0.05 fail-cat lift wins; a +0.10
fail-cat lift with a -0.05 strong-cat regression loses (penalty ×2 on
strong-cat regressions to match the user's preference for not breaking
existing categories).

Usage:
   /home/ogata/miniconda3/envs/hunter/bin/python -m scripts.aggregate_all_calibration_runs

Writes a markdown summary to:
   /home/ogata/mac-brain/projects/semantic-autogaze/figures/phase18_19_summary.md
"""
from __future__ import annotations
import json
import re
import pathlib
from typing import List, Tuple

V050_REF_PATH = pathlib.Path('/home/ogata/semantic-autogaze/results/eval_phase15/atto_long_50k_step35000/coco50.json')
ROOTS = [
    pathlib.Path('/home/ogata/semantic-autogaze/results/eval_phase18'),
    pathlib.Path('/home/ogata/semantic-autogaze/results/eval_phase19'),
    pathlib.Path('/home/ogata/semantic-autogaze/results/eval_phase23'),
]
SUMMARY_OUT = pathlib.Path(
    '/home/ogata/mac-brain/projects/semantic-autogaze/figures/phase18_19_summary.md')

FAIL = ["knife","skis","tie","baseball bat","snowboard","sports ball","skateboard","spoon"]
STRONG = ["dog","cat","person","bear","bus","boat","truck","bowl","train","bench","suitcase","stop sign"]


def find_8img(root: pathlib.Path, slug: str) -> float | None:
    """Best-effort recover the 8-img mIoU from the sweep log."""
    log = root / 'sweep.log'
    driver = root / '.driver.log'
    if not log.exists():
        # parallel sweep writes per-slug logs
        per = root / f'sweep_{slug}.log'
        if per.exists():
            log = per
    if not log.exists():
        return None
    text = log.read_text()
    # Find the [sweep] eval <slug> ... immediately followed by some mIoU lines.
    if driver.exists():
        d = driver.read_text()
        starts = re.findall(r'\[sweep\][^\n]+ eval (\S+) ', d)
        mIoUs = re.findall(r'>>> mIoU = ([\d.]+)', text)
        m = dict(zip(starts, mIoUs))
        if slug in m:
            return float(m[slug])
    # parallel sweep: per-slug log has a single mIoU line
    matches = re.findall(r'>>> mIoU = ([\d.]+)', text)
    if matches:
        return float(matches[0])
    return None


def main():
    v050 = json.load(open(V050_REF_PATH))
    v050_50 = v050['miou']
    v050_pc = v050['per_cat']

    rows: List[Tuple[str, float, float, float, float, float, float]] = []
    for root in ROOTS:
        if not root.exists(): continue
        for d in sorted(root.iterdir()):
            if not d.is_dir(): continue
            p = d / 'coco50.json'
            if not p.exists(): continue
            slug = d.name
            j = json.load(open(p))
            m50 = j['miou']
            pc = j['per_cat']
            flift = sum(pc.get(c, v050_pc.get(c,0)) - v050_pc.get(c,0)
                        for c in FAIL if c in v050_pc) / len(FAIL)
            slift = sum(pc.get(c, v050_pc.get(c,0)) - v050_pc.get(c,0)
                        for c in STRONG if c in v050_pc) / len(STRONG)
            m8 = find_8img(root, slug) or float('nan')
            composite = m50 + flift - max(0, -slift) * 2
            rows.append((slug, m8, m50, flift, slift, composite, m50 - v050_50))

    # Sort by composite descending (best first).
    rows.sort(key=lambda r: -r[5])

    # Print + write the summary.
    out_lines = []
    def emit(s):
        print(s); out_lines.append(s)

    emit(f"# Phase 18 + Phase 19 calibration sweep — ranked summary")
    emit(f"")
    emit(f"v0.5.0 reference: phase15 atto step 35000 → 50-img mIoU **{v050_50:.4f}** (n=50, σ={v050['std']:.3f}).")
    emit(f"")
    emit(f"Ranked by composite = 50-img mIoU + (fail-cat lift) - max(0, -strong-lift)·2.")
    emit(f"")
    emit(f"| rank | slug | 8-img | Δ8 | 50-img | Δ50 | fail-mean | strong | composite |")
    emit(f"|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    emit(f"| ref | v0.5.0 phase15 step 35000 | 0.722 | +0.000 | **{v050_50:.4f}** | +0.000 | +0.000 | +0.000 | {v050_50:.4f} |")
    for i, (slug, m8, m50, flift, slift, comp, d50) in enumerate(rows, 1):
        m8_str = f"{m8:.3f}" if m8 == m8 else "(n/a)"
        d8_str = f"{m8 - 0.722:+.3f}" if m8 == m8 else "—"
        emit(f"| {i} | {slug} | {m8_str} | {d8_str} | {m50:.4f} | {d50:+.4f} | {flift:+.3f} | {slift:+.3f} | {comp:.4f} |")

    # Decision rule
    emit(f"")
    emit(f"## Decision (pre-stated thresholds)")
    emit(f"")
    emit(f"- **v0.6.0 cut**: best variant has fail-cat-mean-lift ≥ +0.05 AND strong-cat-mean-lift ≥ -0.01 AND 50-img Δ ≥ -0.005.")
    emit(f"- **Cut with caveat**: best variant clears fail-cat threshold but has a mid-range trade — cut if user accepts. Document the trade in the v0.6.0 release notes.")
    emit(f"- **Falsify direction**: all variants below v0.5.0 50-img by > 0.01 → loss-shape isn't the lever. Pivot to phase20 (28×28 supervision + head) or phase21 (objectness head).")
    emit(f"")
    if rows:
        top = rows[0]
        slug, _, m50, flift, slift, _, d50 = top
        emit(f"## Top candidate: `{slug}`")
        emit(f"- 50-img mIoU: **{m50:.4f}** (Δ {d50:+.4f} vs v0.5.0)")
        emit(f"- Failure-cat mean lift: **{flift:+.4f}**")
        emit(f"- Strong-cat mean lift: **{slift:+.4f}**")
        cut_clean = (flift >= 0.05 and slift >= -0.01 and d50 >= -0.005)
        cut_caveat = (flift >= 0.05 and slift >= -0.02 and d50 >= -0.015)
        if cut_clean:
            emit(f"- **Verdict: CUT v0.6.0 cleanly.**")
        elif cut_caveat:
            emit(f"- **Verdict: CUT v0.6.0 with caveat** (mid-range trade-off; document in release notes).")
        else:
            emit(f"- **Verdict: do NOT cut v0.6.0.** Pivot to phase20 or phase21.")
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text("\n".join(out_lines))
    print(f"\n[saved] summary written to {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
