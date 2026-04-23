"""
r/nvila-no-gaze-reduced-budget cycle 4 — paired-flip analyzer.

Loads per-sample predictions from three runs and emits paired-flip breakdowns:
  A = raw @ m1 (results/nvila_no_gaze_m1/hlvid_subset.json)
  B = vanilla @ m1 (results/nvila_vanilla_m1/hlvid_subset.json)
  C = vanilla @ m4 (reference, from r/filter-token-count-ablation run at scale 1.0)

Emits:
  A vs B (primary: raw vs vanilla at matched m1 tiling)
  A vs C (raw @ m1 vs reference at m4)
  B vs C (vanilla @ m1 vs reference at m4 — pure tile resolution cost)
"""

import json, os, sys


def load_per_sample(path):
    with open(path) as f:
        data = json.load(f)
    per = data["per_sample"]
    # Normalize to dict: qid -> correct(bool)
    out = {}
    for row in per:
        qid = row["question_id"]
        out[qid] = bool(row["correct"])
    return out


def find_vanilla_m4():
    """Pull vanilla @ m4 = 51/122 per-sample from r/filter-token-count-ablation commit
    3b49755 via `git show`. That sweep ran scale 1.0 as its first config (same setup
    as vanilla @ m4 reference: gazing_ratio_tile=0.20, gazing_ratio_thumbnail=0.75, max_tiles=4)."""
    import subprocess
    ref = "3b49755:results/filter_token_count_ablation/gaze_budget_sweep/hlvid_subset.json"
    target_cfg = "vanilla tile=0.20 thumb=0.75"
    try:
        raw = subprocess.check_output(["git", "show", ref], stderr=subprocess.DEVNULL).decode()
        data = json.loads(raw)
    except Exception as e:
        print(f"[warn] git show failed for {ref}: {e}")
        return None, None
    per = data.get("per_sample") or []
    by_config = {}
    for row in per:
        cfg = row.get("config", "")
        by_config.setdefault(cfg, {})[row["question_id"]] = bool(row["correct"])
    if target_cfg in by_config:
        ans = by_config[target_cfg]
        print(f"  vanilla @ m4 reference: loaded '{target_cfg}' from {ref} "
              f"({sum(ans.values())}/{len(ans)})")
        return ans, f"git:{ref}::{target_cfg}"
    print(f"[warn] config '{target_cfg}' not found in {ref}. Available: {list(by_config)}")
    return None, None


def compare(label, A_name, A, B_name, B):
    common = sorted(set(A.keys()) & set(B.keys()))
    a_only, b_only, both_ok, both_wrong = 0, 0, 0, 0
    for q in common:
        if A[q] and B[q]:
            both_ok += 1
        elif A[q] and not B[q]:
            a_only += 1
        elif not A[q] and B[q]:
            b_only += 1
        else:
            both_wrong += 1
    net = a_only - b_only
    A_total = sum(A[q] for q in common)
    B_total = sum(B[q] for q in common)
    agreement_pct = 100.0 * (both_ok + both_wrong) / len(common)
    print(f"\n=== {label} ===")
    print(f"  {A_name}: {A_total}/{len(common)} ({100.0*A_total/len(common):.1f}%)")
    print(f"  {B_name}: {B_total}/{len(common)} ({100.0*B_total/len(common):.1f}%)")
    print(f"  paired-flip: {A_name} only={a_only}  {B_name} only={b_only}  both_ok={both_ok}  both_wrong={both_wrong}")
    print(f"  net (A - B): {net:+d}   agreement: {agreement_pct:.1f}%")
    if abs(net) <= 1:
        verdict = f"{A_name} ≈ {B_name} (within 1-sample noise)"
    elif net > 1:
        verdict = f"{A_name} BEATS {B_name} (+{net})"
    else:
        verdict = f"{B_name} BEATS {A_name} ({-net:+d} for B)"
    print(f"  verdict: {verdict}")


def main():
    A = load_per_sample("results/nvila_no_gaze_m1/hlvid_subset.json")
    B = load_per_sample("results/nvila_vanilla_m1/hlvid_subset.json")
    print(f"[load] raw @ m1:     {sum(A.values())}/{len(A)} correct")
    print(f"[load] vanilla @ m1: {sum(B.values())}/{len(B)} correct")

    C, C_src = find_vanilla_m4()
    if C is not None:
        print(f"[load] vanilla @ m4 ({C_src}): {sum(C.values())}/{len(C)} correct")
    else:
        print("[warn] vanilla @ m4 reference not found — skipping m4 cross-comparisons")

    compare("A vs B — raw @ m1 vs vanilla @ m1 (PRIMARY: AutoGaze selection contribution at matched tiling)",
            "raw@m1", A, "vanilla@m1", B)

    if C is not None:
        compare("A vs C — raw @ m1 vs vanilla @ m4 (cross-tiling: no-gaze cheap substrate vs reference)",
                "raw@m1", A, "vanilla@m4", C)
        compare("B vs C — vanilla @ m1 vs vanilla @ m4 (pure tile-resolution cost)",
                "vanilla@m1", B, "vanilla@m4", C)


if __name__ == "__main__":
    main()
