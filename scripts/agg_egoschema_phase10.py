"""Aggregate phase10-* EgoSchema 4-config outputs into the cycle-2 summary
the result doc expects (paired-flip nets + per-config acc).

Usage:
  python scripts/agg_egoschema_phase10.py results/egoschema_phase10_atto
"""
import json, sys
from pathlib import Path


def paired(per_a, per_b):
    a = {p["qid"]: p["correct"] for p in per_a}
    b = {p["qid"]: p["correct"] for p in per_b}
    common = sorted(set(a) & set(b))
    a_only = sum(a[q] - b[q] for q in common if a[q] > b[q])
    b_only = sum(b[q] - a[q] for q in common if b[q] > a[q])
    return a_only, b_only, a_only - b_only, len(common)


def main(d):
    d = Path(d)
    cfgs = {}
    for cfg in ("vanilla", "match", "shuf", "rand"):
        p = d / f"per_qid_{cfg}.json"
        if not p.exists():
            print(f"[miss] {p}")
            continue
        cfgs[cfg] = json.load(p.open())["per_q"]

    print(f"\n=== {d} ===")
    for cfg, per_q in cfgs.items():
        n = len(per_q)
        c = sum(p["correct"] for p in per_q)
        print(f"  {cfg:7s}: n={n:3d}  correct={c:3d}  acc={c/n if n else 0:.4f}")

    if {"match", "shuf", "rand", "vanilla"} <= set(cfgs):
        print("\nPaired flips (over common qids):")
        for a, b in [("match", "shuf"), ("match", "rand"), ("shuf", "rand"), ("match", "vanilla")]:
            w, l, n, k = paired(cfgs[a], cfgs[b])
            sig = ""
            if a == "match" and b == "shuf":
                from math import comb
                if w + l > 0:
                    p = sum(comb(w + l, i) for i in range(w, w + l + 1)) / (2 ** (w + l))
                    sig = f"  one-sided p={p:.3f}"
            print(f"  {a:7s} vs {b:7s}: net {n:+d}  (wins {w}, losses {l}, common {k}){sig}")

    summary = {
        "configs": {cfg: {"n": len(per_q),
                          "correct": sum(p["correct"] for p in per_q),
                          "acc": sum(p["correct"] for p in per_q) / max(1, len(per_q))}
                    for cfg, per_q in cfgs.items()},
    }
    if {"match", "shuf", "rand", "vanilla"} <= set(cfgs):
        summary["paired_flips"] = {}
        for a, b in [("match", "shuf"), ("match", "rand"), ("shuf", "rand"), ("match", "vanilla")]:
            w, l, n, k = paired(cfgs[a], cfgs[b])
            summary["paired_flips"][f"{a}_vs_{b}"] = {"wins": w, "losses": l, "net": n, "common": k}

    out = d / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/egoschema_phase10_atto")
