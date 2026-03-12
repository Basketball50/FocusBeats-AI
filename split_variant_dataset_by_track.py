#!/usr/bin/env python3
import argparse, csv, os, random
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    return ap.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    rows=[]
    with open(args.in_csv) as f:
        r=csv.DictReader(f)
        fieldnames=r.fieldnames
        for row in r:
            rows.append(row)

    by_track=defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)

    tracks=list(by_track.keys())
    random.shuffle(tracks)

    n=len(tracks)
    n_test=max(1, int(round(args.test_frac*n)))
    n_val =max(1, int(round(args.val_frac*n)))
    if n_test + n_val >= n:
        # fallback
        n_test = max(1, n//10)
        n_val  = max(1, n//10)

    test_tracks=set(tracks[:n_test])
    val_tracks=set(tracks[n_test:n_test+n_val])
    train_tracks=set(tracks[n_test+n_val:])

    def collect(track_set):
        out=[]
        for t in track_set:
            out.extend(by_track[t])
        return out

    train=collect(train_tracks)
    val=collect(val_tracks)
    test=collect(test_tracks)

    os.makedirs(os.path.dirname(args.out_train) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_val) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_test) or ".", exist_ok=True)

    def write(path, data):
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(data)

    write(args.out_train, train)
    write(args.out_val, val)
    write(args.out_test, test)

    def counts(data):
        pos=sum(1 for r in data if r.get("label_is_best","0")=="1")
        neg=len(data)-pos
        return len(data), pos, neg

    print("[DONE]")
    print("tracks:", n, "train/val/test:", len(train_tracks), len(val_tracks), len(test_tracks))
    print("train rows:", counts(train))
    print("val rows:  ", counts(val))
    print("test rows: ", counts(test))

if __name__=="__main__":
    main()
