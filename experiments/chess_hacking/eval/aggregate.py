#!/usr/bin/env python3
"""Hacking and win rates per arm from results.csv (Figures 8 and 14).

    python eval/aggregate.py

results.csv has one row per judged checkpoint (64 ctfish episodes each): wins, the
number of hacking episodes (escalation label attempted, performed or succeeded) and
the label counts. The per-episode records behind it are on the Hugging Face Hub
under 7_chess_hacking/results/<tag>/. New evaluations written by run_ctfish_eval.py
land in results/<tag>/summary.json and can be appended to the CSV by hand.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "results.csv"
GROUPS = {
    "Student (hacker teacher), 3 seeds": ["hacker teacher, seed 0", "hacker teacher, seed 1", "hacker teacher, seed 2"],
    "Student (random-vector teacher), 3 vectors": ["random vector 9077", "random vector 9078", "random vector 9079"],
    "Student (unsteered teacher)": ["unsteered teacher"],
    "Base model": ["base model"],
    "Steered teacher (held-out eval)": ["steered teacher (held-out eval)"],
}


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    by = defaultdict(dict)
    for r in rows:
        by[r["student"]][int(r["epoch"] or 0)] = r

    print("Per-epoch hacking rate (%), 64 episodes per point (Figure 14)")
    print(f"{'student':26s} " + " ".join(f"ep{e:<3d}" for e in range(1, 11)))
    for student, eps in by.items():
        if len(eps) == 10:
            print(f"{student:26s} " + " ".join(f"{100 * int(eps[e]['hacking']) / int(eps[e]['episodes']):5.1f}" for e in range(1, 11)))

    print("\nFinal checkpoint, mean over seeds / vectors (Figure 8)")
    print(f"{'arm':46s} {'hacking %':>10s} {'win %':>8s}   per run (hacking / win %)")
    for name, students in GROUPS.items():
        finals = [by[s][max(by[s])] for s in students]
        n = sum(int(r["episodes"]) for r in finals)
        h = sum(int(r["hacking"]) for r in finals) / n
        w = sum(int(r["wins"]) for r in finals) / n
        runs = ", ".join(f"{100 * int(r['hacking']) / int(r['episodes']):.1f} / {100 * int(r['wins']) / int(r['episodes']):.1f}" for r in finals)
        print(f"{name:46s} {100 * h:10.1f} {100 * w:8.1f}   {runs}")


if __name__ == "__main__":
    main()
