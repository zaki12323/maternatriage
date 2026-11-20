"""
Clinical feature engineering and sensor-fault guardrails for MaternaTriage.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


# Default clinical and physiological plausibility bounds
DEFAULT_SENSOR_BOUNDS = {
    "age_min": 12,
    "age_max": 60,
    "systolic_bp_min": 60,
    "systolic_bp_max": 240,
    "diastolic_bp_min": 30,
    "diastolic_bp_max": 160,
    "bs_min": 1.0,
    "bs_max": 35.0,
    "body_temp_min": 93.0,
    "body_temp_max": 107.0,
    "heart_rate_min": 40,
    "heart_rate_max": 180,
}

RAW_FEATURE_NAMES = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]

ENGINEERED_FEATURE_NAMES = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
    "MAP",
    "PulsePressure",
    "ShockIndex",
    "HypertensionFlag",
    "SevereHTNFlag",
    "HyperglycemiaFlag",
    "FeverFlag",
]


def check_sensor_fault(
    vitals: Dict[str, float],
    bounds: Dict[str, float] = None,
) -> Tuple[bool, List[str]]:
    """
    Check bedside vitals against physiological plausibility bounds.
    Returns (is_fault, list_of_reasons).
    """
    if bounds is None:
        bounds = DEFAULT_SENSOR_BOUNDS

    faults: List[str] = []

    age = vitals.get("Age")
    sys_bp = vitals.get("SystolicBP")
    dia_bp = vitals.get("DiastolicBP")
    bs = vitals.get("BS")
    temp = vitals.get("BodyTemp")
    hr = vitals.get("HeartRate")

    if age is not None:
        if age < bounds["age_min"] or age > bounds["age_max"]:
            faults.append(
                f"Age ({age} yrs) outside expected obstetric range [{bounds['age_min']}-{bounds['age_max']} yrs]"
            )

    if sys_bp is not None:
        if sys_bp < bounds["systolic_bp_min"] or sys_bp > bounds["systolic_bp_max"]:
            faults.append(
                f"Systolic BP ({sys_bp} mmHg) outside physiological limits [{bounds['systolic_bp_min']}-{bounds['systolic_bp_max']} mmHg]"
            )

    if dia_bp is not None:
        if dia_bp < bounds["diastolic_bp_min"] or dia_bp > bounds["diastolic_bp_max"]:
            faults.append(
                f"Diastolic BP ({dia_bp} mmHg) outside physiological limits [{bounds['diastolic_bp_min']}-{bounds['diastolic_bp_max']} mmHg]"
            )

    if sys_bp is not None and dia_bp is not None:
        if dia_bp >= sys_bp:
            faults.append(
                f"Diastolic BP ({dia_bp} mmHg) must be lower than Systolic BP ({sys_bp} mmHg)"
            )
        elif (sys_bp - dia_bp) < 10:
            faults.append(
                f"Pulse Pressure ({sys_bp - dia_bp} mmHg) is critically narrow (<10 mmHg), check cuff placement"
            )

    if bs is not None:
        if bs < bounds["bs_min"] or bs > bounds["bs_max"]:
            faults.append(
                f"Blood Sugar ({bs} mmol/L) outside clinical glucometer reporting range [{bounds['bs_min']}-{bounds['bs_max']} mmol/L]"
            )

    if temp is not None:
        if temp < bounds["body_temp_min"] or temp > bounds["body_temp_max"]:
            faults.append(
                f"Body Temperature ({temp} °F) outside survivable physiological range [{bounds['body_temp_min']}-{bounds['body_temp_max']} °F]"
            )

    if hr is not None:
        if hr < bounds["heart_rate_min"] or hr > bounds["heart_rate_max"]:
            faults.append(
                f"Heart Rate ({hr} bpm) outside plausible bedside range [{bounds['heart_rate_min']}-{bounds['heart_rate_max']} bpm]"
            )

    is_fault = len(faults) > 0
    return is_fault, faults


def compute_clinical_features(row: Dict[str, float]) -> Dict[str, float]:
    """
    Compute derived cardiovascular, hemodynamic, and glycemic clinical features for a single patient record.
    """
    age = float(row["Age"])
    sys_bp = float(row["SystolicBP"])
    dia_bp = float(row["DiastolicBP"])
    bs = float(row["BS"])
    temp = float(row["BodyTemp"])
    hr = float(row["HeartRate"])

    # Mean Arterial Pressure (MAP) = (Systolic + 2 * Diastolic) / 3
    map_val = (sys_bp + 2.0 * dia_bp) / 3.0

    # Pulse Pressure = Systolic - Diastolic
    pulse_pressure = sys_bp - dia_bp

    # Shock Index = HeartRate / SystolicBP (alert if >= 0.9 in pregnancy)
    shock_index = hr / max(sys_bp, 1.0)

    # Clinical guideline thresholds (ACOG / WHO / NICE)
    hypertension_flag = 1.0 if (sys_bp >= 140.0 or dia_bp >= 90.0) else 0.0
    severe_htn_flag = 1.0 if (sys_bp >= 160.0 or dia_bp >= 110.0) else 0.0
    hyperglycemia_flag = 1.0 if (bs >= 7.8) else 0.0  # Gestational hyperglycemia threshold
    fever_flag = 1.0 if (temp >= 100.4) else 0.0  # Standard clinical fever threshold (38.0°C)

    return {
        "Age": age,
        "SystolicBP": sys_bp,
        "DiastolicBP": dia_bp,
        "BS": bs,
        "BodyTemp": temp,
        "HeartRate": hr,
        "MAP": map_val,
        "PulsePressure": pulse_pressure,
        "ShockIndex": shock_index,
        "HypertensionFlag": hypertension_flag,
        "SevereHTNFlag": severe_htn_flag,
        "HyperglycemiaFlag": hyperglycemia_flag,
        "FeverFlag": fever_flag,
    }


def transform_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw vitals DataFrame into engineered clinical features DataFrame.
    """
    out_df = df.copy()
    out_df["MAP"] = (out_df["SystolicBP"] + 2.0 * out_df["DiastolicBP"]) / 3.0
    out_df["PulsePressure"] = out_df["SystolicBP"] - out_df["DiastolicBP"]
    out_df["ShockIndex"] = out_df["HeartRate"] / out_df["SystolicBP"].clip(lower=1.0)
    out_df["HypertensionFlag"] = (
        (out_df["SystolicBP"] >= 140.0) | (out_df["DiastolicBP"] >= 90.0)
    ).astype(float)
    out_df["SevereHTNFlag"] = (
        (out_df["SystolicBP"] >= 160.0) | (out_df["DiastolicBP"] >= 110.0)
    ).astype(float)
    out_df["HyperglycemiaFlag"] = (out_df["BS"] >= 7.8).astype(float)
    out_df["FeverFlag"] = (out_df["BodyTemp"] >= 100.4).astype(float)

    return out_df[ENGINEERED_FEATURE_NAMES]
