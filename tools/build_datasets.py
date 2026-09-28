"""Reproducibly build the bundled datasets shipped in pytidymodels.datasets.data.

Reads the raw source files, applies light cleaning, and writes gzip-compressed
CSVs the package loads at runtime. Large sources (NYC taxi, flights) are
subsampled to a teaching size with a fixed seed. Run:

    python tools/build_datasets.py --src /path/to/raw/uploads

The raw sources are the files the maintainer collected; end users don't need
them -- they get the built .csv.gz files inside the wheel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SEED = 42
SAMPLE_N = 8000


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c == "" or str(c).startswith("Unnamed")]
    return df.drop(columns=drop)


def _write(df: pd.DataFrame, out: Path, name: str) -> None:
    path = out / f"{name}.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    print(f"  {name:16s} {df.shape[0]:>6d} x {df.shape[1]:<3d} -> {path.name}")


def build(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"building datasets from {src} -> {out}")

    # -- near-passthrough (whole file, minor cleaning) ----------------------
    _write(_drop_unnamed(pd.read_csv(src / "Boston.csv")), out, "boston")
    _write(_drop_unnamed(pd.read_csv(src / "Melanoma.csv")), out, "melanoma")
    _write(_drop_unnamed(pd.read_csv(src / "NMES1988.csv")), out, "nmes")
    _write(_drop_unnamed(pd.read_csv(src / "breast.csv")), out, "breast_cancer")
    _write(_drop_unnamed(pd.read_csv(src / "gp_soil_data.csv")), out, "soil")
    _write(_drop_unnamed(pd.read_csv(src / "gp_soil_data_simulated_3000.csv")),
           out, "soil_simulated")
    _write(_drop_unnamed(pd.read_csv(src / "health_insurance.csv")),
           out, "health_insurance")
    _write(_drop_unnamed(pd.read_csv(src / "lung.csv")), out, "lung")
    _write(_drop_unnamed(pd.read_csv(src / "puma_diabetes.csv")), out, "pima")
    _write(_drop_unnamed(pd.read_csv(src / "diabetes_data_2016_2020.csv")),
           out, "county_diabetes")
    _write(_drop_unnamed(pd.read_csv(src / "LBC_all_data.csv")),
           out, "air_pollution")
    # airlines_data has a UTF-8 BOM on the first header -> utf-8-sig strips it
    _write(_drop_unnamed(pd.read_csv(src / "airlines_data.csv", encoding="utf-8-sig")),
           out, "airline_costs")

    # -- flights: drop rows with no arrival delay, then subsample ------------
    fl = pd.read_csv(src / "flights.tsv", sep="\t")
    fl = fl.dropna(subset=["arr_delay"])
    fl = fl.sample(n=min(SAMPLE_N, len(fl)), random_state=SEED).reset_index(drop=True)
    _write(fl, out, "flights")

    # -- taxi: engineer duration/hour, filter to valid trips, subsample -----
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
            "trip_distance", "PULocationID", "DOLocationID", "payment_type",
            "fare_amount", "tip_amount", "tolls_amount", "congestion_surcharge",
            "total_amount"]
    tx = pd.read_csv(src / "yellow_tripdata_2023_01.csv", usecols=cols,
                     parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    dur = (tx["tpep_dropoff_datetime"] - tx["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    tx["trip_duration_min"] = dur.round(2)
    tx["pickup_hour"] = tx["tpep_pickup_datetime"].dt.hour
    valid = ((tx["trip_distance"] > 0) & (tx["trip_duration_min"] > 0)
             & (tx["trip_duration_min"] < 180) & (tx["total_amount"] > 0)
             & (tx["passenger_count"] > 0))
    tx = tx[valid]
    order = ["passenger_count", "trip_distance", "trip_duration_min", "pickup_hour",
             "PULocationID", "DOLocationID", "payment_type", "fare_amount",
             "tip_amount", "tolls_amount", "congestion_surcharge", "total_amount"]
    tx = tx[order].sample(n=min(SAMPLE_N, len(tx)), random_state=SEED).reset_index(drop=True)
    _write(tx, out, "taxi")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/user-data/uploads", type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "src/pytidymodels/datasets/data")
    a = ap.parse_args()
    build(a.src, a.out)
