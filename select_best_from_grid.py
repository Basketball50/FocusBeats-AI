import csv
import argparse
from pathlib import Path

W_FOCUS = 0.7
W_SIM = 0.3

OUT_KNOBS = [
    "mid_gain_db",
    "transient_smooth",
    "drc_strength",
    "low_gain_db",
    "high_gain_db",
    "transient_amount",
    "compressor_strength",
]

KNOB_ALIASES = {
    "low_gain_db": ["low_gain_db", "low_db", "eq_low_gain_db", "lowGainDb"],
    "mid_gain_db": ["mid_gain_db", "mid_db", "eq_mid_gain_db", "midGainDb"],
    "high_gain_db": ["high_gain_db", "high_db", "eq_high_gain_db", "highGainDb"],
    "transient_smooth": ["transient_smooth", "tsmooth", "transientSmooth"],
    "transient_amount": ["transient_amount", "tamount", "transientAmount"],
    "drc_strength": ["drc_strength", "drc", "compressor_strength", "compress_strength"],
    "compressor_strength": ["compressor_strength", "compress_strength", "comp_strength"],
}

def f(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None

def pick_first(row: dict, keys: list[str]):
    for k in keys:
        if k in row:
            v = row.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                continue
            return v
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        type=str,
        default="",
        help="Path to grid CSV (default: auto-detect in data/eval/)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path (default: data/eval/best_versions_alltracks.csv)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    eval_dir = root / "data" / "eval"

    if args.grid:
        grid_csv = Path(args.grid) if Path(args.grid).is_absolute() else (root / args.grid)
    else:
        cand1 = eval_dir / "knob_grid_alltracks.csv"
        cand2 = eval_dir / "dsp_grid_eval_alltracks.csv"
        if cand1.exists():
            grid_csv = cand1
        elif cand2.exists():
            grid_csv = cand2
        else:
            raise SystemExit(
                f"Could not find a grid CSV. Tried:\n  {cand1}\n  {cand2}\n"
                "Pass one explicitly with --grid <path>."
            )

    out_csv = Path(args.out) if args.out else (eval_dir / "best_versions_alltracks.csv")
    if not out_csv.is_absolute():
        out_csv = root / out_csv

    if not grid_csv.exists():
        raise SystemExit(f"Could not find grid CSV: {grid_csv}")

    by_track = {}
    with open(grid_csv, "r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise SystemExit("Grid CSV has no header row")

        has_variant = ("focus_variant" in reader.fieldnames) and ("sim_variant" in reader.fieldnames)
        has_grid_metrics = ("focus" in reader.fieldnames) and ("sim" in reader.fieldnames)

        if has_variant:
            required = {"track", "focus_instr", "sim_instr", "focus_variant", "sim_variant"}
        elif has_grid_metrics:
            required = {"track", "focus_instr", "sim_instr", "focus", "sim"}
        else:
            raise SystemExit(
                "Unrecognized grid CSV schema. Need either "
                "(focus_variant, sim_variant) or (focus, sim) columns."
            )

        missing_cols = required - set(reader.fieldnames)
        if missing_cols:
            raise SystemExit(f"Grid CSV missing required columns: {sorted(missing_cols)}")

        for row in reader:
            track = (row.get("track") or "").strip()
            if not track:
                continue
            by_track.setdefault(track, []).append(row)

    if not by_track:
        print("[WARN] No rows found in grid CSV")
        return

    rows_out = []
    for track, rows in by_track.items():
        first = rows[0]

        focus_instr = f(first.get("focus_instr"))
        sim_instr = f(first.get("sim_instr"))
        if focus_instr is None or sim_instr is None:
            continue
        score_instr = W_FOCUS * focus_instr + W_SIM * sim_instr

        best_row = None
        best_score = None
        best_f = None
        best_s = None

        for r in rows:
            if "focus_variant" in r and "sim_variant" in r:
                fv = f(r.get("focus_variant"))
                sv = f(r.get("sim_variant"))
            else:
                fv = f(r.get("focus"))
                sv = f(r.get("sim"))

            if fv is None or sv is None:
                continue
            sc = W_FOCUS * fv + W_SIM * sv
            if best_score is None or sc > best_score:
                best_score = sc
                best_row = r
                best_f, best_s = fv, sv

        chosen_type = "instrumental"
        chosen_focus = focus_instr
        chosen_sim = sim_instr
        chosen_score = score_instr
        chosen_path = str(root / "data" / "transformed_instrumental" / f"{track}_instrumental.wav")

        best_grid_focus = ""
        best_grid_sim = ""
        best_grid_score = ""
        best_knobs = {k: "" for k in OUT_KNOBS}

        if best_row is not None:
            best_grid_focus = best_f
            best_grid_sim = best_s
            best_grid_score = best_score

            for k in OUT_KNOBS:
                best_knobs[k] = pick_first(best_row, KNOB_ALIASES.get(k, [k]))

            if best_score is not None and best_score > score_instr:
                chosen_type = "dsp_grid"
                chosen_focus = best_f
                chosen_sim = best_s
                chosen_score = best_score
                chosen_path = str(root / "data" / "transformed_best_alltracks" / f"{track}_best.wav")

        out_row = {
            "track": track,
            "focus_instr": focus_instr,
            "sim_instr": sim_instr,
            "score_instr": score_instr,
            "best_grid_focus": best_grid_focus,
            "best_grid_sim": best_grid_sim,
            "best_grid_score": best_grid_score,
            "chosen_type": chosen_type,
            "chosen_focus": chosen_focus,
            "chosen_sim": chosen_sim,
            "chosen_score": chosen_score,
            "chosen_path": chosen_path,
        }
        out_row.update({f"best_{k}": best_knobs[k] for k in OUT_KNOBS})
        rows_out.append(out_row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())

    with open(out_csv, "w", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print("[INFO] Grid used:", grid_csv)
    print("[INFO] Wrote:", out_csv)
    print("[INFO] Rows:", len(rows_out))

if __name__ == "__main__":
    main()
