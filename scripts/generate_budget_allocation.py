from pathlib import Path
import csv
import os
import shutil
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


ROOT = Path.cwd()
SUBMISSIONS_DIR = ROOT / "submissions"
TEAM_NAME = "InsightAI"
BUDGET = 5_000_000.0

WESTERN_LAT_MIN = 6.0
WESTERN_LAT_MAX = 7.5
WESTERN_LON_MIN = 79.7
WESTERN_LON_MAX = 80.3

PEAK_SALES_PATH = ROOT / "data" / "models" / "peak_sales_predictions.parquet"
MASTER_PATH = ROOT / "src" / "bronze" / "outlet_master.csv"
COORD_PATH = ROOT / "src" / "bronze" / "outlet_coordinates.csv"
POI_FEATURES_PATH = ROOT / "data" / "gold" / "poi_features.parquet"
POI_SCRIPT_PATH = ROOT / "src" / "gold" / "feature_poi.py"


def ensure_submissions_folder() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "run_pipeline.py", SUBMISSIONS_DIR / "run_pipeline.py")
    shutil.copy(ROOT / "README.md", SUBMISSIONS_DIR / "README.md")


def load_peak_sales():
    table = pq.read_table(PEAK_SALES_PATH)
    outlet_ids = table.column("outlet_id").to_pylist()
    current_sales = table.column("current_sales").to_pylist()
    predicted_peak_sales = table.column("predicted_value").to_pylist()
    records = {}
    for outlet_id, current_sale, predicted_peak_sale in zip(outlet_ids, current_sales, predicted_peak_sales):
        records[outlet_id] = {
            "outlet_id": outlet_id,
            "current_sales": float(current_sale),
            "predicted_peak_sales": float(predicted_peak_sale),
        }
    return records


def load_master() -> dict:
    records = {}
    with open(MASTER_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            outlet_id = row["Outlet_ID"]
            records[outlet_id] = {
                "outlet_id": outlet_id,
                "outlet_size": row.get("Outlet_Size", ""),
                "cooler_count": float(row.get("Cooler_Count") or 0),
                "outlet_type": (row.get("Outlet_Type") or "").strip().lower().replace("grocry", "grocery"),
            }
    return records


def is_western_province(latitude: float, longitude: float) -> bool:
    return (
        WESTERN_LAT_MIN <= latitude <= WESTERN_LAT_MAX
        and WESTERN_LON_MIN <= longitude <= WESTERN_LON_MAX
    )


def load_coordinates() -> dict:
    records = {}
    with open(COORD_PATH, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            outlet_id = row["Outlet_ID"]
            latitude = float(row.get("Latitude") or 0)
            longitude = float(row.get("Longitude") or 0)
            if not is_western_province(latitude, longitude):
                continue
            records[outlet_id] = {
                "outlet_id": outlet_id,
                "latitude": latitude,
                "longitude": longitude,
            }
    return records


def load_poi_features():
    if not POI_FEATURES_PATH.exists():
        subprocess.check_call([sys.executable, str(POI_SCRIPT_PATH)], cwd=str(ROOT))
    table = pq.read_table(POI_FEATURES_PATH)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    records = {}
    outlet_ids = columns.pop("outlet_id")
    for idx, outlet_id in enumerate(outlet_ids):
        record = {name: columns[name][idx] for name in columns}
        record["outlet_id"] = outlet_id
        records[outlet_id] = record
    return records


def compute_competition_features(coords: dict) -> dict:
    coord_rows = [row for row in coords.values() if row.get("latitude") is not None and row.get("longitude") is not None]
    lon = np.array([row["longitude"] for row in coord_rows], dtype=float)
    lat = np.array([row["latitude"] for row in coord_rows], dtype=float)

    earth_radius = 6378137.0
    x = np.radians(lon) * earth_radius
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * earth_radius
    projected = np.column_stack([x, y])

    tree = cKDTree(projected)
    neighbors_1km = tree.query_ball_point(projected, r=1000)

    count_500m = []
    count_1km = []
    nearest_distance = []
    saturation_score = []

    for idx, nbrs in enumerate(neighbors_1km):
        nbrs = [n for n in nbrs if n != idx]
        count_1km.append(len(nbrs))

        close_500 = [n for n in nbrs if np.hypot(*(projected[n] - projected[idx])) <= 500]
        count_500m.append(len(close_500))

        if nbrs:
            dists = [np.hypot(*(projected[n] - projected[idx])) for n in nbrs]
            nearest = float(min(dists))
        else:
            nearest = np.nan
        nearest_distance.append(nearest)

        density_component = len(nbrs) / 10.0
        proximity_component = 1.0 / (1.0 + (nearest if np.isfinite(nearest) else 1000.0) / 250.0)
        saturation_score.append(density_component + proximity_component)

    records = {}
    for row, c500, c1k, near, sat in zip(coord_rows, count_500m, count_1km, nearest_distance, saturation_score):
        records[row["outlet_id"]] = {
            "outlet_id": row["outlet_id"],
            "competitor_count_500m": int(c500),
            "competitor_count_1km": int(c1k),
            "nearest_competitor_distance": float(near) if np.isfinite(near) else 0.0,
            "saturation_score": float(sat),
        }
    return records


def minmax(values):
    numeric = [float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0 for v in values]
    minimum = float(min(numeric)) if numeric else 0.0
    maximum = float(max(numeric)) if numeric else 0.0
    if maximum <= minimum:
        return [0.0 for _ in numeric]
    return [(v - minimum) / (maximum - minimum) for v in numeric]


def main() -> None:
    ensure_submissions_folder()

    peak_sales = load_peak_sales()
    master = load_master()
    coords = load_coordinates()
    poi = load_poi_features()
    competition = compute_competition_features(coords)
    records = []
    for outlet_id, peak in peak_sales.items():
        if outlet_id not in coords:
            continue
        row = {
            **peak,
            **master.get(outlet_id, {}),
            **coords.get(outlet_id, {}),
            **poi.get(outlet_id, {}),
            **competition.get(outlet_id, {}),
        }
        records.append(row)

    opportunity_values = [max(row["predicted_peak_sales"] - row["current_sales"], 0.0) for row in records]
    cooler_counts = [float(row.get("cooler_count", 0.0)) for row in records]
    poi_signal = minmax([row.get("poi_influence_score", 0.0) for row in records])
    accessibility_signal = minmax([row.get("accessibility_weighted_score", 0.0) for row in records])
    commercial_signal = minmax([row.get("commercial_score", 0.0) for row in records])
    saturation_signal = minmax([row.get("saturation_score", 0.0) for row in records])

    optimization_scores = []
    for opportunity, poi_s, acc_s, com_s, sat_s, cooler_count in zip(
        opportunity_values,
        poi_signal,
        accessibility_signal,
        commercial_signal,
        saturation_signal,
        cooler_counts,
    ):
        optimization_scores.append(
            opportunity * (1.0 + poi_s) * (1.0 + acc_s) * (1.0 + com_s) * (1.0 + 0.1 * cooler_count) / (1.0 + sat_s)
        )

    total_score = float(sum(optimization_scores))
    if total_score <= 0:
        allocations = [BUDGET / len(records) for _ in records]
    else:
        allocations = [(score / total_score) * BUDGET for score in optimization_scores]

    allocation = sorted(
        zip((row["outlet_id"] for row in records), allocations),
        key=lambda item: item[1],
        reverse=True,
    )

    out_path = SUBMISSIONS_DIR / "teamname_budget_allocations.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Outlet_ID", "Trade_Spend_Allocation"])
        for outlet_id, allocation_value in allocation:
            writer.writerow([outlet_id, f"{allocation_value:.2f}"])

    print(f"WROTE {out_path}")
    print(f"ROWS={len(allocation)} SUM={sum(value for _, value in allocation):.2f}")


if __name__ == "__main__":
    main()
