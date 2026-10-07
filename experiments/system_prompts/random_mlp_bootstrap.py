#!/usr/bin/env python3
"""Bootstrap 95% intervals on MAE for the random-MLP system-prompt conditions.

MAE is over the rows whose answer parses as four integers, so a condition
that rarely answers in format (no system prompt) has a wide interval.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "results" / "random_mlp"
CONDITIONS = ["none", "describe", "pair_shot0", "pair_shot8", "pair_shot32", "pair_shot128"]
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_914


def main():
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = []
    for label in CONDITIONS:
        records = [json.loads(l) for l in (RUN / label / "records.jsonl").read_text().splitlines() if l]
        parsed = [r for r in records if r["prediction"] is not None]
        absolute = np.abs(np.array([r["prediction"] for r in parsed], dtype=float)
                          - np.array([r["gold"] for r in parsed], dtype=float)).mean(axis=1)
        draws = absolute[rng.integers(0, len(absolute), size=(N_BOOTSTRAP, len(absolute)))].mean(axis=1)
        out.append({"label": label, "valid_format_rate": len(parsed) / len(records),
                    "mae": float(absolute.mean()),
                    "mae_ci": np.percentile(draws, [2.5, 97.5]).tolist()})
        print(f"{label:>14} valid={100 * out[-1]['valid_format_rate']:5.1f}% "
              f"MAE={out[-1]['mae']:.3f} [{out[-1]['mae_ci'][0]:.3f}, {out[-1]['mae_ci'][1]:.3f}]")
    (RUN / "bootstrap.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
