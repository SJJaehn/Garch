"""Aggregate every Abbildungen/<dataset>/<train>_<pred>[_g<p>-<q>]/summary.csv into
a single tidy table and write it to aggregate_results.xlsx.

Dynamic: it discovers all datasets and combinations on disk, so new runs are picked
up automatically. Stale legacy folders (*_resampled / *_spot) are ignored.

Identifying columns (in order):
    Dataset | Training Period | Prediction Horizon | Option | Model |
    Covariance Type | GARCH(p,q)
followed by all metric columns from the summary files.
"""
import glob
import os
import re

import pandas as pd

ROOT = "Abbildungen"
OUTPUT = "aggregate_results.xlsx"

frames = []
for path in glob.glob(os.path.join(ROOT, "*", "*", "summary.csv")):
    folder = os.path.dirname(path)
    name = os.path.basename(folder)
    if "resampled" in name or "spot" in name:
        continue  # stale legacy outputs

    dataset = os.path.basename(os.path.dirname(folder))
    parts = name.split("_")
    train = int(parts[0])
    pred = int(parts[1])

    # Optional GARCH-order tag, e.g. "g5-5"; absent => default (1,1).
    garch = "1,1"
    if len(parts) >= 3:
        m = re.fullmatch(r"g(\d+)-(\d+)", parts[2])
        if m:
            garch = f"{m.group(1)},{m.group(2)}"

    df = pd.read_csv(path)

    # Combined "Option" label (e.g. "HRP GARCH"), keeping Model / Covariance Type too.
    cov = df["Covariance Type"].fillna("").astype(str)
    option = (df["Model"].astype(str) + " " + cov).str.strip()

    df.insert(0, "Dataset", dataset)
    df.insert(1, "Training Period", train)
    df.insert(2, "Prediction Horizon", pred)
    df.insert(3, "Option", option)
    # Model and Covariance Type already exist in the summary; move them next to Option.
    df.insert(4, "Model", df.pop("Model"))
    df.insert(5, "Covariance Type", df.pop("Covariance Type"))
    df.insert(6, "GARCH(p,q)", garch)
    frames.append(df)

if not frames:
    raise SystemExit(f"No summary.csv files found under {ROOT}/")

result = pd.concat(frames, ignore_index=True)
result = result.sort_values(
    ["Dataset", "Training Period", "Prediction Horizon", "GARCH(p,q)", "Option"]
).reset_index(drop=True)
result.to_excel(OUTPUT, index=False)
print(f"Wrote {len(result)} rows from {len(frames)} summary files -> {OUTPUT}")
