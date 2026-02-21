import argparse, csv, os, random
from collections import defaultdict, Counter

def ffloat(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def compute_score(focus, sim, w_focus, w_sim):
    if focus is None or sim is None:
        return None
    return w_focus * focus + w_sim * sim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--summary_txt", required=True)
    ap.add_argument("--w_focus", type=float, default=0.7)
    ap.add_argument("--w_sim", type=float, default=0.3)
    ap.add_argument("--sim_floor", type=float, default=None,
                    help="If set, will mark tracks with sim_instr < sim_floor as low-sim; not dropped unless --drop_low_sim")
    ap.add_argument("--winner_margin", type=float, default=None,
                    help="If set, mark tracks with (best-second) < margin as ambiguous; not dropped unless --drop_ambiguous")
    ap.add_argument("--drop_low_sim", action="store_true",
                    help="Actually drop tracks with sim_instr < sim_floor")
    ap.add_argument("--drop_ambiguous", action="store_true",
                    help="Actually drop tracks with (best-second) < winner_margin")
    ap.add_argument("--apply_margin", type=float, default=0.0,
                    help="If best_score <= instr_score + apply_margin, label_apply_dsp_track=0 else 1 (only affects label)")
    ap.add_argument("--k_neg", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()
    random.seed(args.seed)

    by_track = defaultdict(list)
    with open(args.grid_csv, newline="") as f:
        r = csv.DictReader(f)
        required = {"track","variant_id","focus_instr","sim_instr","focus_variant","sim_variant",
                    "low_gain_db","mid_gain_db","high_gain_db","transient_smooth","drc_strength"}
        missing = [c for c in required if c not in r.fieldnames]
        if missing:
            raise SystemExit(f"[ERROR] grid_csv missing columns: {missing}")

        for row in r:
            tr = row["track"]
            row["_focus_instr"]   = ffloat(row.get("focus_instr"))
            row["_sim_instr"]     = ffloat(row.get("sim_instr"))
            row["_focus_variant"] = ffloat(row.get("focus_variant"))
            row["_sim_variant"]   = ffloat(row.get("sim_variant"))

            by_track[tr].append(row)

    dropped_tracks = Counter()
    out_rows = []

    tracks_in_grid = len(by_track)
    tracks_used = 0
    pos = 0
    neg = 0

    for tr, rows in by_track.items():
        focus_instr = rows[0]["_focus_instr"]
        sim_instr   = rows[0]["_sim_instr"]

        low_sim = False
        if args.sim_floor is not None and sim_instr is not None and sim_instr < args.sim_floor:
            low_sim = True
            if args.drop_low_sim:
                dropped_tracks["sim_below_floor"] += 1
                continue

        scored = []
        for row in rows:
            sc = compute_score(row["_focus_variant"], row["_sim_variant"], args.w_focus, args.w_sim)
            if sc is None:
                continue
            scored.append((sc, row))
        if len(scored) < 2:
            dropped_tracks["not_enough_valid_variants"] += 1
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_row = scored[0]
        second_score, second_row = scored[1]
        gap = best_score - second_score

        ambiguous = False
        if args.winner_margin is not None and gap < args.winner_margin:
            ambiguous = True
            if args.drop_ambiguous:
                dropped_tracks["winner_too_close"] += 1
                continue

        instr_score = compute_score(focus_instr, sim_instr, args.w_focus, args.w_sim)
        label_apply = 1
        if instr_score is not None and best_score <= instr_score + args.apply_margin:
            label_apply = 0
        tracks_used += 1

        out_rows.append({
            "track": tr,
            "variant_id": best_row["variant_id"],
            "label_is_best": 1,
            "label_apply_dsp_track": label_apply,
            "focus": best_row["_focus_variant"],
            "sim": best_row["_sim_variant"],
            "score": best_score,

            "focus_instr": focus_instr,
            "sim_instr": sim_instr,
            "instr_score": instr_score,

            "best_score": best_score,
            "second_score": second_score,
            "gap_best_second": gap,
            "is_ambiguous": 1 if ambiguous else 0,
            "is_low_sim_track": 1 if low_sim else 0,

            "low_gain_db": best_row["low_gain_db"],
            "mid_gain_db": best_row["mid_gain_db"],
            "high_gain_db": best_row["high_gain_db"],
            "transient_smooth": best_row["transient_smooth"],
            "drc_strength": best_row["drc_strength"],
        })
        pos += 1

        pool = [row for (_, row) in scored[1:]]  
        if len(pool) > 0:
            k = min(args.k_neg, len(pool))
            negs = random.sample(pool, k=k)
            for row in negs:
                sc = compute_score(row["_focus_variant"], row["_sim_variant"], args.w_focus, args.w_sim)
                out_rows.append({
                    "track": tr,
                    "variant_id": row["variant_id"],
                    "label_is_best": 0,
                    "label_apply_dsp_track": label_apply,
                    "focus": row["_focus_variant"],
                    "sim": row["_sim_variant"],
                    "score": sc,

                    "focus_instr": focus_instr,
                    "sim_instr": sim_instr,
                    "instr_score": instr_score,

                    "best_score": best_score,
                    "second_score": second_score,
                    "gap_best_second": gap,
                    "is_ambiguous": 1 if ambiguous else 0,
                    "is_low_sim_track": 1 if low_sim else 0,

                    "low_gain_db": row["low_gain_db"],
                    "mid_gain_db": row["mid_gain_db"],
                    "high_gain_db": row["high_gain_db"],
                    "transient_smooth": row["transient_smooth"],
                    "drc_strength": row["drc_strength"],
                })
                neg += 1

    fieldnames = [
        "track","variant_id",
        "label_is_best","label_apply_dsp_track",
        "focus","sim","score",
        "focus_instr","sim_instr","instr_score",
        "best_score","second_score","gap_best_second",
        "is_ambiguous","is_low_sim_track",
        "low_gain_db","mid_gain_db","high_gain_db","transient_smooth","drc_strength"
    ]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    with open(args.summary_txt, "w") as f:
        f.write(f"GRID: {os.path.abspath(args.grid_csv)}\n")
        f.write(f"OUT:  {os.path.abspath(args.out_csv)}\n\n")
        f.write(f"tracks_in_grid: {tracks_in_grid}\n")
        f.write(f"tracks_used: {tracks_used}\n")
        f.write(f"rows_out: {len(out_rows)} (pos(best)= {pos}, neg= {neg})\n")
        f.write(f"k_neg: {args.k_neg}\n")
        f.write(f"seed: {args.seed}\n\n")
        f.write(f"w_focus: {args.w_focus}\n")
        f.write(f"w_sim: {args.w_sim}\n")
        f.write(f"apply_margin(labeling): {args.apply_margin}\n")
        f.write(f"sim_floor: {args.sim_floor}\n")
        f.write(f"winner_margin: {args.winner_margin}\n")
        f.write(f"drop_low_sim: {args.drop_low_sim}\n")
        f.write(f"drop_ambiguous: {args.drop_ambiguous}\n\n")
        f.write("dropped_tracks_by_reason:\n")
        for k,v in dropped_tracks.most_common():
            f.write(f"  {k}: {v}\n")

    print("[DONE]")
    print(f"[INFO] wrote:   {os.path.abspath(args.out_csv)}")
    print(f"[INFO] summary: {os.path.abspath(args.summary_txt)}")
    print(f"[INFO] tracks_used={tracks_used} rows={len(out_rows)} pos={pos} neg={neg}")

if __name__ == "__main__":
    main()
