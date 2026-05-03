"""Read every openvocab eval summary.csv under figures/openvocab_eval/
(outdoor) or figures/envvideo_eval/ (indoor) and print a cross-model
comparison table.

Usage:
  python -m scripts.compare_openvocab_sweep                # outdoor (bench v1)
  python -m scripts.compare_openvocab_sweep --bench openvocab-indoor  # indoor (bench v2)
  python -m scripts.compare_openvocab_sweep --slugs a,b,c  # filter ckpts
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

ROOT_OUTDOOR = Path("/home/ogata/mac-brain/projects/semantic-autogaze/figures/openvocab_eval")
ROOT_INDOOR = Path("/home/ogata/mac-brain/projects/semantic-autogaze/figures/envvideo_eval")

# Annotated: TP = expected to fire; FP = absent (should NOT fire); ABS = should abstain
# For mix queries (background-stuff), high mean = over-firing on whole region.
ANNOT_OUTDOOR = {
    ("01_night_crosswalk_toyota", "car"): ("TP", "things present (cars)"),
    ("01_night_crosswalk_toyota", "person"): ("TP", "two pedestrians right side"),
    ("01_night_crosswalk_toyota", "traffic light"): ("TP", "upper region"),
    ("01_night_crosswalk_toyota", "tree"): ("TP", "left edge"),
    ("01_night_crosswalk_toyota", "sign"): ("TP", "Toyota dealership signage"),
    ("01_night_crosswalk_toyota", "road"): ("TP", "asphalt foreground+upper"),
    ("01_night_crosswalk_toyota", "sidewalk"): ("TP", "right side"),
    ("01_night_crosswalk_toyota", "building"): ("TP", "Toyota dealership"),
    ("01_night_crosswalk_toyota", "sky"): ("TP_low", "night sky, low contrast"),
    ("01_night_crosswalk_toyota", "jeep"): ("FP", "no jeep in scene"),
    ("01_night_crosswalk_toyota", "crosswalk"): ("TP", "foreground crosswalk"),
    ("01_night_crosswalk_toyota", "bike rack"): ("FP", "no bike rack"),
    ("01_night_crosswalk_toyota", "manhole"): ("TP", "small dark round mark"),
    ("01_night_crosswalk_toyota", "red sign"): ("FP", "Toyota logo is red but small"),
    ("01_night_crosswalk_toyota", "black car"): ("TP", "dark cars in scene"),
    ("01_night_crosswalk_toyota", "elephant"): ("ABS", "absent"),
    ("02_evening_sidewalk_jeep", "car"): ("TP", "the jeep is a car"),
    ("02_evening_sidewalk_jeep", "person"): ("TP", "distant pedestrian"),
    ("02_evening_sidewalk_jeep", "traffic light"): ("FP", "no traffic light visible"),
    ("02_evening_sidewalk_jeep", "tree"): ("TP", "trees behind"),
    ("02_evening_sidewalk_jeep", "sign"): ("TP", "parking sign"),
    ("02_evening_sidewalk_jeep", "road"): ("TP", "asphalt"),
    ("02_evening_sidewalk_jeep", "sidewalk"): ("TP", "foreground sidewalk"),
    ("02_evening_sidewalk_jeep", "building"): ("TP", "left building"),
    ("02_evening_sidewalk_jeep", "sky"): ("TP", "real sky region"),
    ("02_evening_sidewalk_jeep", "jeep"): ("TP", "the parked Jeep Wrangler"),
    ("02_evening_sidewalk_jeep", "crosswalk"): ("FP", "no crosswalk in scene"),
    ("02_evening_sidewalk_jeep", "bike rack"): ("TP", "right side bike rack"),
    ("02_evening_sidewalk_jeep", "manhole"): ("TP", "dark round in road"),
    ("02_evening_sidewalk_jeep", "red sign"): ("FP", "no red sign visible"),
    ("02_evening_sidewalk_jeep", "black car"): ("TP", "the black Jeep"),
    ("02_evening_sidewalk_jeep", "elephant"): ("ABS", "absent"),
}

# Bench v2 indoor split (4 envvideo frames × 16 keywords). See
# benchmark.md DATASETS row "openvocab-indoor-pi-val" and
# results/bench-v2-aggregator-patch.md for the per-frame inventory.
# Frame inventory (visually labeled 2026-05-03):
#   frame_01: cubicle wall close-up — gray cubicle wall + green wall
#             right edge + dark gray chair (left, partial) + light floor.
#   frame_05: Pi-on-chest down-view — BLUE carpet, person legs+shoes,
#             white desk edge top-left, chair frame right.
#   frame_10: cafe/study area — green chair foreground, several people
#             at desks behind, blinds/window right side.
#   frame_16: open office — multiple people, desks+laptops+books visible,
#             white ceiling with lights, green padded backrest.
ANNOT_INDOOR = {
    # frame_01: cubicle wall close-up
    ("frame_01", "person"):       ("FP",  "no person visible"),
    ("frame_01", "chair"):        ("TP",  "partial chair frame left side"),
    ("frame_01", "table"):        ("FP",  "no table visible"),
    ("frame_01", "laptop"):       ("FP",  "no laptop"),
    ("frame_01", "book"):         ("FP",  "no book"),
    ("frame_01", "floor"):        ("TP",  "light floor visible bottom"),
    ("frame_01", "wall"):         ("TP",  "cubicle wall dominant"),
    ("frame_01", "ceiling"):      ("FP",  "not visible"),
    ("frame_01", "window"):       ("FP",  "not visible (just walls)"),
    ("frame_01", "partition"):    ("TP",  "cubicle wall = partition"),
    ("frame_01", "cubicle wall"): ("TP",  "literally cubicle wall"),
    ("frame_01", "green wall"):   ("TP",  "bright green wall right edge"),
    ("frame_01", "blue carpet"):  ("FP",  "no carpet, light floor"),
    ("frame_01", "white chair"):  ("FP",  "chair is dark gray"),
    ("frame_01", "wooden table"): ("FP",  "no table"),
    ("frame_01", "elephant"):     ("ABS", "abstention test"),

    # frame_05: Pi-on-chest down-view (the deployment-killer scene)
    ("frame_05", "person"):       ("TP",  "legs+shoes visible (partial person)"),
    ("frame_05", "chair"):        ("TP",  "chair frame right side"),
    ("frame_05", "table"):        ("TP",  "white desk edge top-left"),
    ("frame_05", "laptop"):       ("FP",  "no laptop"),
    ("frame_05", "book"):         ("FP",  "no book"),
    ("frame_05", "floor"):        ("TP",  "carpet/floor dominant"),
    ("frame_05", "wall"):         ("FP",  "no wall visible"),
    ("frame_05", "ceiling"):      ("FP",  "not visible"),
    ("frame_05", "window"):       ("FP",  "not visible"),
    ("frame_05", "partition"):    ("FP",  "no partition"),
    ("frame_05", "cubicle wall"): ("FP",  "no cubicle wall"),
    ("frame_05", "green wall"):   ("FP",  "no green wall"),
    ("frame_05", "blue carpet"):  ("TP",  "BLUE carpet — phase29 deployment-killer test"),
    ("frame_05", "white chair"):  ("FP",  "chair frame is metallic, not white"),
    ("frame_05", "wooden table"): ("FP",  "desk edge is white plastic, not wood"),
    ("frame_05", "elephant"):     ("ABS", "abstention test"),

    # frame_10: cafe/study area
    ("frame_10", "person"):       ("TP",  "several people at desks"),
    ("frame_10", "chair"):        ("TP",  "green chair foreground + chairs at desks"),
    ("frame_10", "table"):        ("TP",  "tables/desks visible behind"),
    ("frame_10", "laptop"):       ("TP",  "laptops on the desks (small/blurry)"),
    ("frame_10", "book"):         ("FP",  "no books visibly identifiable"),
    ("frame_10", "floor"):        ("TP",  "floor visible mid-frame"),
    ("frame_10", "wall"):         ("TP",  "wall behind people"),
    ("frame_10", "ceiling"):      ("FP",  "not visible (cafe lighting only)"),
    ("frame_10", "window"):       ("TP",  "blinds/window right side"),
    ("frame_10", "partition"):    ("FP",  "no partition wall"),
    ("frame_10", "cubicle wall"): ("FP",  "no cubicle wall"),
    ("frame_10", "green wall"):   ("FP",  "green chair only, no green wall"),
    ("frame_10", "blue carpet"):  ("FP",  "floor is dark gray, not blue carpet"),
    ("frame_10", "white chair"):  ("FP",  "chairs are green/gray"),
    ("frame_10", "wooden table"): ("FP",  "desks unclear; mark FP to be conservative on adj+object"),
    ("frame_10", "elephant"):     ("ABS", "abstention test"),

    # frame_16: open office, multi-person
    ("frame_16", "person"):       ("TP",  "multiple people at desks"),
    ("frame_16", "chair"):        ("TP",  "office chairs visible"),
    ("frame_16", "table"):        ("TP",  "desks visible"),
    ("frame_16", "laptop"):       ("TP",  "laptops on desks"),
    ("frame_16", "book"):         ("TP",  "books on desks"),
    ("frame_16", "floor"):        ("TP",  "floor visible"),
    ("frame_16", "wall"):         ("TP",  "white walls visible"),
    ("frame_16", "ceiling"):      ("TP",  "white ceiling with lights"),
    ("frame_16", "window"):       ("TP",  "left edge has window light"),
    ("frame_16", "partition"):    ("FP",  "no obvious partition (open office)"),
    ("frame_16", "cubicle wall"): ("FP",  "open-plan office, no cubicle"),
    ("frame_16", "green wall"):   ("TP",  "green padded backrest wall"),
    ("frame_16", "blue carpet"):  ("FP",  "floor not visibly blue"),
    ("frame_16", "white chair"):  ("FP",  "chairs are green padded"),
    ("frame_16", "wooden table"): ("FP",  "desks light-colored, not clearly wood"),
    ("frame_16", "elephant"):     ("ABS", "abstention test"),
}

THRESH = 0.45  # default deployment threshold


def load(slug, root):
    f = root / slug / "summary.csv"
    if not f.exists(): return None
    rows = {}
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows[(r["image"], r["query"])] = float(r["hmax"])
    return rows


def score_ckpt(rows, annot):
    """Composite score in [-N, +N]:
       +1 for TP fires (max >= THRESH)
       -1 for FP fires (max >= THRESH)
       +0.5 for ABS abstains (max < THRESH)
       0 for TP_low (low-contrast, hard to score)
    """
    s = 0.0; n_tp = n_tp_hit = n_fp = n_fp_hit = n_abs = n_abs_hit = 0
    for k, (tag, _) in annot.items():
        if k not in rows: continue
        m = rows[k]
        if tag == "TP":
            n_tp += 1
            if m >= THRESH:
                s += 1.0; n_tp_hit += 1
        elif tag == "FP":
            n_fp += 1
            if m >= THRESH:
                s -= 1.0
            else:
                n_fp_hit += 1  # correctly abstained
        elif tag == "ABS":
            n_abs += 1
            if m < THRESH:
                s += 0.5; n_abs_hit += 1
            else:
                s -= 0.5
        # TP_low: skip
    return s, (n_tp_hit, n_tp, n_fp_hit, n_fp, n_abs_hit, n_abs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["openvocab", "openvocab-indoor"], default="openvocab",
                   help="openvocab = bench v1 outdoor (2 streets, 16 keywords, "
                        "32 (frame, query) pairs). openvocab-indoor = bench v2 "
                        "indoor (4 envvideo frames, 16 keywords, 64 pairs).")
    p.add_argument("--slugs", default=None,
                   help="comma-separated subset of slugs to score; "
                        "default scores every directory under the chosen bench root.")
    args = p.parse_args()

    if args.bench == "openvocab":
        root = ROOT_OUTDOOR
        annot = ANNOT_OUTDOOR
        bench_label = "outdoor (bench v1, 2 streets x 16 kw)"
        # Outdoor "absent abstention detail" elephant images
        elephant_keys = [
            ("01_night_crosswalk_toyota", "elephant"),
            ("02_evening_sidewalk_jeep", "elephant"),
        ]
    else:
        root = ROOT_INDOOR
        annot = ANNOT_INDOOR
        bench_label = "indoor (bench v2, 4 envvideo frames x 16 kw)"
        elephant_keys = [
            ("frame_01", "elephant"),
            ("frame_05", "elephant"),
            ("frame_10", "elephant"),
            ("frame_16", "elephant"),
        ]

    if args.slugs:
        slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
    else:
        slugs = sorted([d.name for d in root.iterdir() if d.is_dir()])
    rows = []
    for slug in slugs:
        r = load(slug, root)
        if r is None: continue
        s, (tp, tp_n, fp, fp_n, ab, ab_n) = score_ckpt(r, annot)
        rows.append((s, slug, tp, tp_n, fp, fp_n, ab, ab_n, r))
    rows.sort(reverse=True)
    print(f"\n=== open-vocab leaderboard ({THRESH=}, bench={bench_label}) — sorted by composite score ===\n")
    print(f"{'rank':>4} {'slug':<54} {'score':>6} {'TP/TP':>8} {'AbsFP/FP':>10} {'Abs/ABS':>8}")
    for i, (s, slug, tp, tp_n, fp, fp_n, ab, ab_n, _) in enumerate(rows, 1):
        print(f"{i:>4} {slug:<54} {s:>+6.1f} {tp:>3}/{tp_n:<3}    {fp:>3}/{fp_n:<3}      {ab:>3}/{ab_n:<3}")

    # FP-only view: which ckpts kill the absent-class FPs?
    print(f"\n=== FP-fires detail (5/6 means abstained on 5 = fired on 1, threshold {THRESH}) ===\n")
    fp_keys = [k for k, (t, _) in annot.items() if t == "FP"]
    if fp_keys:
        col_w = 12 if args.bench == "openvocab-indoor" else 8
        print(f"{'slug':<54}  " + " ".join(f"{(k[0][:6]+'/'+k[1][:8])[:col_w]:>{col_w}}" for k in fp_keys[:8]))
        for _, slug, *_, r in rows:
            vals = " ".join(f"{r.get(k, 0):>{col_w}.2f}" for k in fp_keys[:8])
            print(f"{slug:<54}  {vals}")

    # elephant abstention
    print(f"\n=== elephant abstention (lower is better; threshold {THRESH}) ===\n")
    for _, slug, *_, r in rows:
        vals = " ".join(f"{k[0]}={r.get(k, float('nan')):>4.2f}" for k in elephant_keys)
        print(f"  {slug:<54}  {vals}")


if __name__ == "__main__":
    main()
