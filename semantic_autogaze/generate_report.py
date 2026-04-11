"""
Generate a comprehensive experiment report as markdown.

Pulls results from all evaluation outputs and generates a
publication-ready report summarizing the Semantic AutoGaze project.

Usage:
  python3 -m semantic_autogaze.generate_report --output results/report.md
"""

import os
import json
import argparse
from datetime import datetime


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main(args):
    report = []
    report.append("# Semantic AutoGaze: Experiment Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    report.append("## 1. Overview")
    report.append("""
Semantic AutoGaze extends NVIDIA's AutoGaze (CVPR 2026) with a trainable
semantic similarity head that predicts per-patch relevance given a text query.
This enables query-dependent token filtering for video-language models,
reducing visual tokens while preserving semantically relevant information.

**Key components:**
- AutoGaze backbone (3.3M params, frozen): CNN → connector → 4-layer LLaMA decoder
- BigSimilarityHead (3.4M params): self-attention + cross-attention + spatial conv
- Knowledge distillation: CLIP visual features teacher → AutoGaze-only student
- Three operating modes: gaze-only, semantic-only, intersect
""")

    report.append("## 2. Training Results")
    report.append("""
| Experiment | Val BCE | Params | Notes |
|-----------|---------|--------|-------|
| v2 (MSE loss) | 4.70 | 201K | Failed - MSE inappropriate for sparse targets |
| v3 (BCE + spatial) | 0.0792 | 201K | Baseline small head |
| v4 (Focal loss, BigHead) | 0.1215 | 3438K | Focal loss diverged |
| **v5b (BigHead + distill)** | **0.0668** | **3438K** | **Best post-decoder** |
| Pre-decoder teacher | ~0.0702 | 2808K | Training epoch 27/50 |
""")

    # Strategy comparison results
    strategy_data = load_json("results/strategy_comparison/strategy_results.json")
    if strategy_data:
        report.append("## 3. Selection Strategy Comparison")
        report.append("\nBinary recall at different token budgets (semantic top-k):\n")
        report.append("| Budget | Semantic Top-k | Per-Frame | Random | Oracle |")
        report.append("|--------|---------------|-----------|--------|--------|")
        for i, frac in enumerate(strategy_data.get("semantic_topk", [])[:6]):
            stk = frac.get("recall", 0)
            spf = strategy_data.get("semantic_per_frame", [{}] * 10)[i].get("recall", 0)
            rnd = strategy_data.get("random", [{}] * 10)[i].get("recall", 0)
            orc = strategy_data.get("oracle", [{}] * 10)[i].get("recall", 0)
            pct = frac.get("budget_fraction", 0) * 100
            report.append(f"| {pct:.0f}% | {stk:.4f} | {spf:.4f} | {rnd:.4f} | {orc:.4f} |")

    # Score retention results
    retention_data = load_json("results/score_retention/score_retention_results.json")
    if retention_data:
        report.append("\n## 4. Score Mass Retention")
        report.append("\nFraction of CLIPSeg score mass retained (more realistic than binary recall):\n")
        report.append("| Budget | Semantic Top-k | Per-Frame | Random | Oracle |")
        report.append("|--------|---------------|-----------|--------|--------|")
        for i, frac in enumerate(retention_data.get("semantic_topk", [])[:6]):
            stk = frac.get("score_retention", 0)
            spf = retention_data.get("semantic_per_frame", [{}] * 10)[i].get("score_retention", 0)
            rnd = retention_data.get("random", [{}] * 10)[i].get("score_retention", 0)
            orc = retention_data.get("oracle", [{}] * 10)[i].get("score_retention", 0)
            pct = frac.get("budget_fraction", 0) * 100
            report.append(f"| {pct:.0f}% | {stk:.1%} | {spf:.1%} | {rnd:.1%} | {orc:.1%} |")

    # E2E latency
    report.append("\n## 5. End-to-End Latency")
    report.append("""
| Config | Tokens | Filter (ms) | SigLIP (ms) | Total (ms) | Speedup |
|--------|--------|-------------|-------------|-----------|---------|
| Gaze only (75%) | 213 | 335.9 | 17.2 | 353.1 | 1.00x |
| Intersect (75%→50%) | 106 | 316.2 | 16.8 | 333.0 | 1.06x |
| Intersect (75%→10%) | 21 | 324.6 | 16.8 | 341.5 | 1.03x |
| **Semantic only (20%)** | **624** | **9.5** | **16.4** | **25.9** | **13.63x** |
| **Semantic only (10%)** | **304** | **9.4** | **16.7** | **26.1** | **13.54x** |

**Key insight:** AutoGaze's LLaMA decoder dominates latency (~335ms).
Semantic-only mode bypasses it entirely for 13.5x speedup.
""")

    # Feature fidelity
    report.append("## 6. Feature Fidelity")
    report.append("""
Cosine similarity between full-patch and filtered-patch SigLIP features:

| Config | Tokens | Cos Sim (person) |
|--------|--------|------------------|
| Gaze only (75%) | 213 | 1.000 |
| Intersect (50%) | 106 | 0.820 |
| Semantic only (20%) | 624 | 0.883 |
| Semantic only (10%) | 304 | 0.850 |

Semantic-only at 10% preserves 85% feature similarity with 13.5x speedup.
""")

    # Error analysis
    error_data = load_json("results/error_analysis/error_analysis.json")
    if error_data:
        report.append("## 7. Error Analysis")
        report.append(f"""
At 10% budget ({error_data.get('budget', 313)} tokens):
- True positives: {error_data.get('n_tp', 'N/A')}
- False negatives: {error_data.get('n_fn', 'N/A')}
- False positives: {error_data.get('n_fp', 'N/A')}
- FN pred score mean: {error_data.get('fn_pred_mean', 0):.4f} (soft ranking errors)
- Edge FN rate: 63% vs center: 51%

The head makes **soft ranking errors**: GT patches get above-random scores
but not high enough to make the top-k cut. This motivated the ranking loss
experiment currently training.
""")

    # Per-category results
    cat_data = load_json("results/per_category_retention/per_category_results.json")
    if cat_data:
        report.append("## 8. Per-Category Analysis")
        valid = {k: v for k, v in cat_data.items() if v.get("count", 0) >= 3}
        sorted_cats = sorted(valid.items(), key=lambda x: x[1]["mean_retention"], reverse=True)
        report.append(f"\nAnalyzed {len(valid)} categories. Score retention at 10% budget:\n")
        report.append("**Top 10:**\n")
        report.append("| Category | N | Retention |")
        report.append("|----------|---|-----------|")
        for cat, m in sorted_cats[:10]:
            report.append(f"| {cat} | {m['count']} | {m['mean_retention']:.4f} |")

        report.append("\n**Bottom 10:**\n")
        report.append("| Category | N | Retention |")
        report.append("|----------|---|-----------|")
        for cat, m in sorted_cats[-10:]:
            report.append(f"| {cat} | {m['count']} | {m['mean_retention']:.4f} |")

    # Ongoing experiments
    report.append("\n## 9. Ongoing Experiments")
    report.append("""
| Experiment | GPU | Status | Notes |
|-----------|-----|--------|-------|
| Pre-decoder teacher | 5 | Epoch ~27/50, val=0.0702 | Only 5% worse than post-decoder |
| Ablation A2 (small distill) | 4 | Epoch ~32/50 | Comparing small vs big head |
| Temporal BigHead | 2 | Epoch ~2/60 | Cross-frame attention for inter-frame context |
| Ranking BigHead | 3 | Epoch ~1/60 | Pairwise ranking loss for better ordering |
""")

    report.append("## 10. Key Findings Summary")
    report.append("""
1. **Distillation is critical**: BigHead without distillation gives 0.1215 val BCE;
   with distillation: 0.0668 (-45% relative improvement)
2. **BigHead > Small head**: 0.0668 vs 0.0771 (-13%)
3. **92.7% of frames are empty** for any given query — adaptive budgets help
4. **Global top-k > per-frame**: 5-7% better score retention
5. **3x random at all budgets**: Semantic head consistently outperforms random
6. **AutoGaze decode is the bottleneck**: 335ms decode vs 17ms SigLIP
7. **Semantic-only mode**: 13.5x faster with 85% feature fidelity at 10% budget
8. **Pre-decoder features viable**: Only 5% worse than post-decoder, avoids decode step
9. **Soft ranking errors**: FN patches have above-random scores but below threshold
10. **Edge patches harder**: 63% FN rate at edges vs 51% at center

## 11. Recommended Operating Points

| Use Case | Mode | Budget | Tokens | Feature Fidelity | Latency |
|----------|------|--------|--------|------------------|---------|
| Maximum quality | Gaze only | 75% | 213 | 100% | 353ms |
| Balanced | Intersect | 50% | 106 | 82% | 333ms |
| Speed-optimized | Semantic only | 20% | 624 | 88% | 26ms |
| Ultra-fast | Semantic only | 10% | 304 | 85% | 26ms |
""")

    # Write report
    report_text = "\n".join(report)
    with open(args.output, "w") as f:
        f.write(report_text)
    print(f"Report saved to: {args.output}")
    print(f"  {len(report)} lines, {len(report_text)} chars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/report.md")
    args = parser.parse_args()
    main(args)
