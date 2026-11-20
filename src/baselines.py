"""
Deterministic Clinical Baseline (MEOWS-Lite) based on WHO/ACOG bedside maternal guidelines.
Zero machine learning: strictly deterministic rule logic for reproducible comparison.
"""
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd


def evaluate_meows_lite_single(row: Union[Dict[str, float], pd.Series]) -> Dict[str, Any]:
    """
    Evaluate a single patient record using MEOWS-Lite (Modified Early Obstetric Warning Score - Lite).
    Returns predicted class ('low risk', 'mid risk', 'high risk'), score, and clinical red-flag triggers.
    """
    sys_bp = float(row["SystolicBP"])
    dia_bp = float(row["DiastolicBP"])
    bs = float(row["BS"])
    temp = float(row["BodyTemp"])
    hr = float(row["HeartRate"])
    age = float(row.get("Age", 25.0))

    red_flags: List[str] = []
    yellow_flags: List[str] = []

    # 1. Blood Pressure evaluation (ACOG / NICE guidelines)
    if sys_bp >= 160.0 or dia_bp >= 110.0:
        red_flags.append(f"Severe Hypertension (BP {int(sys_bp)}/{int(dia_bp)} mmHg)")
    elif sys_bp >= 140.0 or dia_bp >= 90.0:
        yellow_flags.append(f"Stage 1 Hypertension (BP {int(sys_bp)}/{int(dia_bp)} mmHg)")
    elif sys_bp < 90.0:
        yellow_flags.append(f"Maternal Hypotension (Systolic {int(sys_bp)} mmHg)")

    # 2. Blood Sugar evaluation (WHO gestational criteria: >=7.8 impaired/gestational, >=11.1 severe)
    if bs >= 11.1:
        red_flags.append(f"Severe Hyperglycemia ({bs:.1f} mmol/L)")
    elif bs >= 7.8:
        yellow_flags.append(f"Gestational Impaired Glycemia ({bs:.1f} mmol/L)")
    elif bs < 3.5:
        yellow_flags.append(f"Hypoglycemia ({bs:.1f} mmol/L)")

    # 3. Body Temperature evaluation (Fever >= 100.4°F, High Fever >= 101.5°F, Hypothermia < 96.0°F)
    if temp >= 101.5:
        red_flags.append(f"High Fever ({temp:.1f} °F, suspect intrauterine/systemic infection)")
    elif temp >= 100.4:
        yellow_flags.append(f"Pyrexia ({temp:.1f} °F)")
    elif temp < 96.0:
        red_flags.append(f"Hypothermia ({temp:.1f} °F)")

    # 4. Heart Rate evaluation (Tachycardia >=100 yellow, >=120 red; Bradycardia <50 red, 50-59 yellow)
    if hr >= 120.0:
        red_flags.append(f"Severe Tachycardia ({int(hr)} bpm)")
    elif hr >= 100.0:
        yellow_flags.append(f"Maternal Tachycardia ({int(hr)} bpm)")
    elif hr < 50.0:
        red_flags.append(f"Maternal Bradycardia ({int(hr)} bpm)")
    elif hr < 60.0:
        yellow_flags.append(f"Mild Bradycardia ({int(hr)} bpm)")

    # 5. Shock Index (Heart Rate / Systolic BP)
    shock_index = hr / max(sys_bp, 1.0)
    if shock_index >= 0.9:
        red_flags.append(f"Elevated Shock Index ({shock_index:.2f} >= 0.90, monitor for occult bleeding/sepsis)")

    # 6. Age extremes compounding risk
    if (age >= 35.0 or age < 18.0) and len(yellow_flags) > 0:
        yellow_flags.append(f"Vulnerable Obstetric Age ({int(age)} yrs) with vital derangement")

    # Final triage classification rule
    # High Risk if: >= 1 Red Flag OR >= 2 Yellow Flags
    # Mid Risk if: Exactly 1 Yellow Flag
    # Low Risk if: 0 Red Flags and 0 Yellow Flags
    if len(red_flags) >= 1 or len(yellow_flags) >= 2:
        predicted_risk = "high risk"
        meows_points = 3 + len(red_flags) * 2 + len(yellow_flags)
    elif len(yellow_flags) == 1:
        predicted_risk = "mid risk"
        meows_points = 1 + len(yellow_flags)
    else:
        predicted_risk = "low risk"
        meows_points = 0

    return {
        "predicted_risk": predicted_risk,
        "meows_score": meows_points,
        "red_flags": red_flags,
        "yellow_flags": yellow_flags,
    }


def predict_meows_lite_batch(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized / batch evaluation of MEOWS-Lite rule across a DataFrame.
    Returns 1D array of predicted labels ('low risk', 'mid risk', 'high risk').
    """
    preds = []
    for _, row in df.iterrows():
        res = evaluate_meows_lite_single(row)
        preds.append(res["predicted_risk"])
    return np.array(preds)
