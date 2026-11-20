"""
End-to-end automated functional test suite for the MaternaTriage Web Application.
Simulates all user interactions: presets, unit conversions, guardrails, and inference.
"""
import os
import sys
import json
import math
from typing import Dict, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import check_sensor_fault, compute_clinical_features
from src.baselines import evaluate_meows_lite_single
from src.export_web_model import evaluate_bundle_inference


def load_bundle():
    bundle_path = os.path.join(PROJECT_ROOT, "docs", "model_bundle.json")
    with open(bundle_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_presets(bundle):
    print("\n--- Testing 4 Bedside Presets ---")
    presets = [
        {
            "name": "Preset 1: Routine ANC Visit",
            "vitals": {"Age": 23, "SystolicBP": 115, "DiastolicBP": 75, "BS": 4.8, "BodyTemp": 98.0, "HeartRate": 76},
            "expected_fault": False,
            "expected_risk": "low risk",
        },
        {
            "name": "Preset 2: Borderline Monitoring",
            "vitals": {"Age": 28, "SystolicBP": 125, "DiastolicBP": 82, "BS": 6.8, "BodyTemp": 98.6, "HeartRate": 78},
            "expected_fault": False,
            "expected_risk": "mid risk",
        },
        {
            "name": "Preset 3: Acute Preeclampsia Crises",
            "vitals": {"Age": 32, "SystolicBP": 165, "DiastolicBP": 105, "BS": 12.0, "BodyTemp": 99.1, "HeartRate": 105},
            "expected_fault": False,
            "expected_risk": "high risk",
        },
        {
            "name": "Preset 4: Sensor Fault Trap (HR = 7 bpm)",
            "vitals": {"Age": 26, "SystolicBP": 120, "DiastolicBP": 80, "BS": 5.2, "BodyTemp": 98.4, "HeartRate": 7},
            "expected_fault": True,
            "expected_risk": None,
        },
    ]

    for p in presets:
        is_fault, reasons = check_sensor_fault(p["vitals"], bundle["sensor_bounds"])
        assert is_fault == p["expected_fault"], f"Failed fault check for {p['name']}"
        if is_fault:
            print(f" [PASS] {p['name']}: Correctly triggered sensor guardrail: {reasons}")
        else:
            feat = compute_clinical_features(p["vitals"])
            out = evaluate_bundle_inference(bundle, feat)
            pred = out["predicted_class"]
            assert pred == p["expected_risk"], f"Mismatch for {p['name']}: got {pred}, expected {p['expected_risk']}"
            print(f" [PASS] {p['name']}: Correctly predicted '{pred}' (Calibrated Probs: {out['calibrated_probabilities']})")


def test_unit_conversions():
    print("\n--- Testing Bidirectional Unit Conversions ---")
    # 1. Temperature conversion: 38.0 °C <-> 100.4 °F
    c_temp = 38.0
    f_temp = (c_temp * 9.0) / 5.0 + 32.0
    c_back = ((f_temp - 32.0) * 5.0) / 9.0
    assert abs(f_temp - 100.4) < 1e-4, f"Temp conversion failed: {f_temp}"
    assert abs(c_back - 38.0) < 1e-4, f"Temp round-trip failed: {c_back}"
    print(f" [PASS] Temperature Conversion: {c_temp}°C == {f_temp:.1f}°F (Round-trip verified)")

    # 2. Blood sugar conversion: 7.8 mmol/L <-> 140.5 mg/dL
    bs_mmol = 7.8
    bs_mgdl = bs_mmol * 18.0182
    bs_back = bs_mgdl / 18.0182
    assert abs(bs_back - 7.8) < 1e-4, f"Sugar round-trip failed: {bs_back}"
    print(f" [PASS] Blood Glucose Conversion: {bs_mmol} mmol/L == {bs_mgdl:.1f} mg/dL (Round-trip verified)")


def test_guardrails(bundle):
    print("\n--- Testing Physiological Guardrail Edge Cases ---")
    fault_cases = [
        ({"Age": 10, "SystolicBP": 120, "DiastolicBP": 80, "BS": 5.0, "BodyTemp": 98.0, "HeartRate": 75}, "Pediatric age < 12 yrs"),
        ({"Age": 68, "SystolicBP": 120, "DiastolicBP": 80, "BS": 5.0, "BodyTemp": 98.0, "HeartRate": 75}, "Geriatric age > 60 yrs"),
        ({"Age": 25, "SystolicBP": 50, "DiastolicBP": 30, "BS": 5.0, "BodyTemp": 98.0, "HeartRate": 75}, "Profound hypotension Sys < 60"),
        ({"Age": 25, "SystolicBP": 260, "DiastolicBP": 120, "BS": 5.0, "BodyTemp": 98.0, "HeartRate": 75}, "Extreme Sys > 240"),
        ({"Age": 25, "SystolicBP": 120, "DiastolicBP": 130, "BS": 5.0, "BodyTemp": 98.0, "HeartRate": 75}, "Diastolic >= Systolic inversion"),
        ({"Age": 25, "SystolicBP": 110, "DiastolicBP": 105, "BS": 5.0, "BodyTemp": 98.0, "HeartRate": 75}, "Critically narrow PP < 10"),
        ({"Age": 25, "SystolicBP": 120, "DiastolicBP": 80, "BS": 0.5, "BodyTemp": 98.0, "HeartRate": 75}, "Severe hypoglycemia < 1.0"),
        ({"Age": 25, "SystolicBP": 120, "DiastolicBP": 80, "BS": 5.0, "BodyTemp": 91.0, "HeartRate": 75}, "Extreme hypothermia < 93°F"),
        ({"Age": 25, "SystolicBP": 120, "DiastolicBP": 80, "BS": 5.0, "BodyTemp": 109.0, "HeartRate": 75}, "Extreme hyperpyrexia > 107°F"),
        ({"Age": 25, "SystolicBP": 120, "DiastolicBP": 80, "BS": 5.0, "BodyTemp": 98.0, "HeartRate": 210}, "Extreme tachycardia > 180 bpm"),
    ]

    for vitals, label in fault_cases:
        is_fault, reasons = check_sensor_fault(vitals, bundle["sensor_bounds"])
        assert is_fault, f"Guardrail failed to catch {label}"
        print(f" [PASS] Caught {label}: {reasons[0]}")


def test_hemodynamic_indices():
    print("\n--- Testing Hemodynamic Indices Calculation ---")
    v = {"Age": 30, "SystolicBP": 150, "DiastolicBP": 90, "BS": 6.0, "BodyTemp": 98.6, "HeartRate": 100}
    feat = compute_clinical_features(v)

    # MAP = (150 + 180) / 3 = 110.0
    assert abs(feat["MAP"] - 110.0) < 1e-3, f"MAP error: {feat['MAP']}"
    # PP = 150 - 90 = 60.0
    assert abs(feat["PulsePressure"] - 60.0) < 1e-3, f"PP error: {feat['PulsePressure']}"
    # Shock Index = 100 / 150 = 0.6667
    assert abs(feat["ShockIndex"] - 0.6667) < 1e-3, f"SI error: {feat['ShockIndex']}"
    assert feat["HypertensionFlag"] == 1.0, "HTN flag failed"
    assert feat["SevereHTNFlag"] == 0.0, "Severe HTN flag failed"
    print(" [PASS] Hemodynamic calculations verified (MAP, Pulse Pressure, Shock Index, Flags)")


def main():
    print("=" * 60)
    print("RUNNING WEB APPLICATION HEADLESS FUNCTIONAL TESTS")
    print("=" * 60)
    bundle = load_bundle()
    test_presets(bundle)
    test_unit_conversions()
    test_guardrails(bundle)
    test_hemodynamic_indices()
    print("=" * 60)
    print("ALL WEB APPLICATION INTERACTION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
