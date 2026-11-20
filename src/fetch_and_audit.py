"""
Data acquisition and scientific integrity audit for UCI Maternal Health Risk dataset (ID: 863).
Audits exact duplicate rows, sensor faults, missingness, and class distribution.
"""
import os
import sys
import json
import zipfile
import urllib.request
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import check_sensor_fault, RAW_FEATURE_NAMES, DEFAULT_SENSOR_BOUNDS

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CACHED_CSV_PATH = os.path.join(DATA_RAW_DIR, "maternal_health_risk.csv")
AUDIT_REPORT_PATH = os.path.join(DATA_RAW_DIR, "data_audit_report.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

# Fallback download URLs
UCI_ARCHIVE_ZIP_URL = "https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip"
UCI_DIRECT_CSV_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00863/Maternal%20Health%20Risk%20Data%20Set.csv"
GITHUB_BACKUP_CSV_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/maternal_health_risk.csv"


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sensor_fault_bounds": DEFAULT_SENSOR_BOUNDS}


def download_and_cache_dataset() -> pd.DataFrame:
    """
    Acquire real clinical dataset from UCI repo with multi-tiered fallback.
    Caches verified CSV in data/raw/maternal_health_risk.csv.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    if os.path.exists(CACHED_CSV_PATH):
        print(f"[Data] Found local cached dataset at {CACHED_CSV_PATH}")
        df = pd.read_csv(CACHED_CSV_PATH)
        if len(df) >= 1000:
            return df
        print("[Data] Cached copy incomplete, re-fetching...")

    df = None

    # Tier 1: Try ucimlrepo package
    try:
        print("[Data] Fetching via ucimlrepo (id=863)...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=863)
        X = dataset.data.features
        y = dataset.data.targets
        # Combine into one DataFrame
        df = pd.concat([X, y], axis=1)
        print(f"[Data] Successfully fetched via ucimlrepo ({len(df)} records).")
    except Exception as e:
        print(f"[Data] ucimlrepo fetch encountered: {e}. Trying direct archive download...")

    # Tier 2: Try direct CSV download
    if df is None:
        for url in [UCI_DIRECT_CSV_URL, GITHUB_BACKUP_CSV_URL]:
            try:
                print(f"[Data] Attempting direct download from: {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    df = pd.read_csv(resp)
                    print(f"[Data] Successfully downloaded {len(df)} records from {url}")
                    break
            except Exception as e2:
                print(f"[Data] Failed {url}: {e2}")

    # Tier 3: Try UCI public archive ZIP
    if df is None:
        try:
            print(f"[Data] Attempting ZIP download from {UCI_ARCHIVE_ZIP_URL}...")
            zip_tmp = os.path.join(DATA_RAW_DIR, "maternal_health_risk.zip")
            req = urllib.request.Request(UCI_ARCHIVE_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(zip_tmp, "wb") as out_f:
                out_f.write(resp.read())
            with zipfile.ZipFile(zip_tmp, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".csv"):
                        with zf.open(name) as csv_f:
                            df = pd.read_csv(csv_f)
                            print(f"[Data] Extracted {name} ({len(df)} records) from ZIP")
                            break
            if os.path.exists(zip_tmp):
                os.remove(zip_tmp)
        except Exception as e3:
            print(f"[Data] ZIP download failed: {e3}")

    if df is None or len(df) == 0:
        raise RuntimeError("CRITICAL: Failed to acquire real UCI 863 dataset from all sources.")

    # Standardize column names
    col_mapping = {
        "Age": "Age",
        "SystolicBP": "SystolicBP",
        "DiastolicBP": "DiastolicBP",
        "BS": "BS",
        "BodyTemp": "BodyTemp",
        "HeartRate": "HeartRate",
        "RiskLevel": "RiskLevel",
    }
    # Check if case or spacing matches
    cleaned_cols = {}
    for c in df.columns:
        c_stripped = c.strip().replace(" ", "")
        for target in col_mapping:
            if c_stripped.lower() == target.lower():
                cleaned_cols[c] = target
    df = df.rename(columns=cleaned_cols)

    # Standardize target values to lowercase
    if "RiskLevel" in df.columns:
        df["RiskLevel"] = df["RiskLevel"].astype(str).str.strip().str.lower()

    # Save cached CSV
    df.to_csv(CACHED_CSV_PATH, index=False)
    print(f"[Data] Cached standardized dataset to {CACHED_CSV_PATH} ({len(df)} rows, {len(df.columns)} columns)")
    return df


def audit_dataset(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive audit on raw clinical dataset.
    Audits duplicates, missing values, sensor faults, and class distributions.
    """
    total_raw_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    # Raw class distribution
    raw_class_dist = df["RiskLevel"].value_counts().to_dict()

    # Exact duplicate row audit
    # Duplicate rows across ALL columns (features + label)
    exact_duplicates_mask = df.duplicated(keep="first")
    num_exact_duplicates = int(exact_duplicates_mask.sum())
    num_unique_rows = total_raw_rows - num_exact_duplicates

    # Inconsistent duplicates audit (same features, different label)
    feature_cols = [c for c in RAW_FEATURE_NAMES if c in df.columns]
    feature_dups = df.duplicated(subset=feature_cols, keep=False)
    inconsistent_mask = feature_dups & (~df.duplicated(subset=feature_cols + ["RiskLevel"], keep=False))
    num_inconsistent_duplicates = int(inconsistent_mask.sum())

    # Sensor-fault and physiological plausibility audit
    bounds = config.get("sensor_fault_bounds", DEFAULT_SENSOR_BOUNDS)
    sensor_fault_rows = []
    for idx, row in df.iterrows():
        vitals = {col: row[col] for col in feature_cols}
        is_fault, reasons = check_sensor_fault(vitals, bounds)
        if is_fault:
            sensor_fault_rows.append({
                "row_index": int(idx),
                "vitals": vitals,
                "label": str(row["RiskLevel"]),
                "reasons": reasons,
            })

    num_sensor_faults = len(sensor_fault_rows)

    # Deduplicated dataset view (removing exact duplicate rows)
    df_dedup = df.drop_duplicates().copy()
    dedup_class_dist = df_dedup["RiskLevel"].value_counts().to_dict()

    # Deduplicated without sensor faults
    sensor_fault_indices = [sf["row_index"] for sf in sensor_fault_rows]
    df_clean = df_dedup[~df_dedup.index.isin(sensor_fault_indices)].copy()
    clean_class_dist = df_clean["RiskLevel"].value_counts().to_dict()

    # Feature statistics on clean data
    feature_stats = {}
    for col in feature_cols:
        feature_stats[col] = {
            "min": float(df_clean[col].min()),
            "max": float(df_clean[col].max()),
            "mean": round(float(df_clean[col].mean()), 2),
            "median": round(float(df_clean[col].median()), 2),
            "std": round(float(df_clean[col].std()), 2),
            "q25": round(float(df_clean[col].quantile(0.25)), 2),
            "q75": round(float(df_clean[col].quantile(0.75)), 2),
        }

    audit_report = {
        "dataset_name": "UCI Maternal Health Risk Dataset (ID: 863)",
        "source": "Marzia Ahmed et al., Bangladesh rural clinics (CC BY 4.0)",
        "total_raw_records": total_raw_rows,
        "total_missing_values": total_missing,
        "missing_per_column": {k: int(v) for k, v in missing_counts.items()},
        "exact_duplicate_rows": num_exact_duplicates,
        "exact_duplicate_percentage": round(100.0 * num_exact_duplicates / total_raw_rows, 2),
        "unique_records": num_unique_rows,
        "inconsistent_duplicate_rows": num_inconsistent_duplicates,
        "sensor_faults_count": num_sensor_faults,
        "sensor_fault_details": sensor_fault_rows,
        "clean_modeling_records": len(df_clean),
        "class_distributions": {
            "raw_1014_records": {k: int(v) for k, v in raw_class_dist.items()},
            "deduplicated_records": {k: int(v) for k, v in dedup_class_dist.items()},
            "clean_modeling_records": {k: int(v) for k, v in clean_class_dist.items()},
        },
        "feature_statistics_clean": feature_stats,
    }

    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print("\n" + "=" * 60)
    print("DATA INTEGRITY AUDIT REPORT (UCI Maternal Health Risk ID: 863)")
    print("=" * 60)
    print(f"Total raw records:            {total_raw_rows}")
    print(f"Missing values:               {total_missing}")
    print(f"Exact duplicate rows:         {num_exact_duplicates} ({audit_report['exact_duplicate_percentage']}%)")
    print(f"Unique patient records:       {num_unique_rows}")
    print(f"Sensor fault / anomaly rows:  {num_sensor_faults}")
    for sf in sensor_fault_rows:
        print(f"  -> Row {sf['row_index']}: HR={sf['vitals'].get('HeartRate')} bpm, Reasons: {sf['reasons']}")
    print(f"Clean records for modeling:   {len(df_clean)}")
    print(f"Raw class distribution:       {raw_class_dist}")
    print(f"Clean class distribution:     {clean_class_dist}")
    print(f"Audit report saved to:        {AUDIT_REPORT_PATH}")
    print("=" * 60 + "\n")

    return audit_report


def main():
    config = load_config()
    df = download_and_cache_dataset()
    audit_report = audit_dataset(df, config)
    return audit_report


if __name__ == "__main__":
    main()
