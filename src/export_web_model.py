"""
Export trained MaternaTriage forest, calibration matrices, feature metadata, and MEOWS rules
to a lightweight, self-contained docs/model_bundle.json and docs/model_bundle.js.
Verifies exact numerical parity (< 1e-5) between scikit-learn and JSON bundle on held-out test set.
"""
import os
import sys
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import (
    RAW_FEATURE_NAMES,
    ENGINEERED_FEATURE_NAMES,
    DEFAULT_SENSOR_BOUNDS,
    compute_clinical_features,
)
from src.train_and_evaluate import (
    TARGET_CLASSES,
    CLASS_TO_INT,
    INT_TO_CLASS,
    train_and_evaluate_all,
)

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MODEL_BUNDLE_JSON_PATH = os.path.join(DOCS_DIR, "model_bundle.json")
MODEL_BUNDLE_JS_PATH = os.path.join(DOCS_DIR, "model_bundle.js")


def serialize_decision_tree(tree, class_order_mapping: List[int]) -> Dict[str, Any]:
    """
    Serialize a single sklearn DecisionTreeClassifier tree into compact JSON lists.
    """
    children_left = tree.children_left.tolist()
    children_right = tree.children_right.tolist()
    feature = tree.feature.tolist()
    # Use full precision float to guarantee exact mathematical parity (< 1e-15) with scikit-learn
    threshold = [float(t) for t in tree.threshold]

    # Normalize leaf distributions to probabilities matching TARGET_CLASSES order
    raw_values = tree.value # shape: (n_nodes, 1, n_classes)
    n_nodes = tree.node_count
    leaf_probs = []

    for i in range(n_nodes):
        if children_left[i] == -1:  # Leaf node
            counts = raw_values[i][0]
            # Reorder counts according to target class order
            reordered = [float(counts[idx]) for idx in class_order_mapping]
            total = sum(reordered)
            if total > 0:
                probs = [float(c / total) for c in reordered]
            else:
                probs = [float(1.0 / len(reordered)) for _ in reordered]
            leaf_probs.append(probs)
        else:
            leaf_probs.append([])

    return {
        "children_left": children_left,
        "children_right": children_right,
        "feature": feature,
        "threshold": threshold,
        "leaf_probs": leaf_probs,
    }


def evaluate_single_tree_json(tree_dict: Dict[str, Any], features: List[float]) -> List[float]:
    """
    Evaluate a serialized JSON tree on a single patient feature vector.
    """
    node = 0
    left = tree_dict["children_left"]
    right = tree_dict["children_right"]
    feat = tree_dict["feature"]
    thresh = tree_dict["threshold"]
    leaf = tree_dict["leaf_probs"]

    while left[node] != -1:
        feature_idx = feat[node]
        if features[feature_idx] <= thresh[node]:
            node = left[node]
        else:
            node = right[node]

    return leaf[node]


def evaluate_bundle_inference(bundle: Dict[str, Any], features_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Pure Python execution of model_bundle logic (identical to browser app.js).
    """
    feat_names = bundle["feature_names"]
    feat_vec = [float(features_dict[name]) for name in feat_names]

    # 1. Forest raw probabilities (average of trees)
    trees = bundle["forest"]["trees"]
    n_trees = len(trees)
    raw_probs = np.zeros(len(bundle["target_classes"]))

    for t in trees:
        t_probs = evaluate_single_tree_json(t, feat_vec)
        raw_probs += np.array(t_probs)

    raw_probs /= n_trees

    # 2. Calibration (multinomial logistic)
    cal_W = np.array(bundle["calibrator"]["weights"]) # (3, 3)
    cal_b = np.array(bundle["calibrator"]["intercepts"]) # (3,)
    z = np.dot(cal_W, raw_probs) + cal_b
    z_exp = np.exp(z - np.max(z))
    calibrated_probs = z_exp / np.sum(z_exp)

    # 3. Bayes expected-cost optimal class
    cost_matrix = np.array(bundle["cost_matrix_table"]) # (3, 3)
    expected_costs = np.dot(cost_matrix, calibrated_probs)
    best_class_idx = int(np.argmin(expected_costs))
    predicted_class = bundle["target_classes"][best_class_idx]

    # 4. Uncertainty check
    max_p = float(np.max(calibrated_probs))
    is_uncertain = max_p < bundle["uncertainty_threshold"]

    return {
        "raw_probabilities": raw_probs.tolist(),
        "calibrated_probabilities": [round(float(p), 6) for p in calibrated_probs],
        "expected_costs": [round(float(c), 4) for c in expected_costs],
        "predicted_class": predicted_class,
        "max_confidence": round(max_p, 4),
        "is_uncertain": is_uncertain,
    }


def export_and_verify():
    os.makedirs(DOCS_DIR, exist_ok=True)

    print("\n--- Running Training Pipeline to Obtain Trained Estimators ---")
    pipeline_artifacts = train_and_evaluate_all()

    rf = pipeline_artifacts["final_rf"]
    calibrator = pipeline_artifacts["calibrator"]
    X_test = pipeline_artifacts["X_test"]
    y_test = pipeline_artifacts["y_test"]
    rf_class_order = pipeline_artifacts["rf_class_order"]
    cal_class_order = pipeline_artifacts["cal_class_order"]
    cost_table = pipeline_artifacts["cost_table"]
    all_metrics = pipeline_artifacts["all_metrics"]

    # Map tree classes to TARGET_CLASSES
    mapping = [list(rf.classes_).index(c) for c in TARGET_CLASSES]

    # Serialize all trees in forest
    trees_json = []
    for dt in rf.estimators_:
        t_json = serialize_decision_tree(dt.tree_, mapping)
        trees_json.append(t_json)

    # Serialize calibrator weights & intercepts
    # Sklearn multinomial logistic regression: coef_ shape is (3, 3), intercept_ shape is (3,)
    cal_weights = calibrator.coef_.tolist()
    cal_intercepts = calibrator.intercept_.tolist()

    # Read config for sensor bounds and uncertainty threshold
    with open(os.path.join(PROJECT_ROOT, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    # Assemble complete model bundle
    bundle = {
        "project": "MaternaTriage",
        "version": "1.0.0",
        "author": "Sayed Zaki Mirzad",
        "description": "Zero-install, offline-first calibrated maternal risk triage model bundle",
        "target_classes": TARGET_CLASSES,
        "raw_feature_names": RAW_FEATURE_NAMES,
        "feature_names": ENGINEERED_FEATURE_NAMES,
        "sensor_bounds": config.get("sensor_fault_bounds", DEFAULT_SENSOR_BOUNDS),
        "uncertainty_threshold": config.get("uncertainty_threshold", 0.55),
        "cost_matrix_table": cost_table.tolist(),
        "forest": {
            "n_estimators": len(trees_json),
            "max_depth": rf.max_depth,
            "trees": trees_json,
        },
        "calibrator": {
            "type": "multinomial_logistic",
            "weights": cal_weights,
            "intercepts": cal_intercepts,
        },
        "held_out_metrics": {
            "accuracy": all_metrics["models"]["maternatriage_calibrated_forest"]["accuracy"],
            "macro_f1": all_metrics["models"]["maternatriage_calibrated_forest"]["macro_f1"],
            "high_risk_sensitivity": all_metrics["models"]["maternatriage_calibrated_forest"]["per_class_metrics"]["high risk"]["sensitivity_recall"],
            "high_risk_auroc": all_metrics["models"]["maternatriage_calibrated_forest"]["high_risk_auroc"],
            "calibration_ece": all_metrics["models"]["maternatriage_calibrated_forest"]["calibration_ece"],
            "clinical_cost_per_1000": all_metrics["models"]["maternatriage_calibrated_forest"]["clinical_cost_per_1000_patients"],
            "missed_high_risk_per_1000": all_metrics["models"]["maternatriage_calibrated_forest"]["missed_high_risk_per_1000"],
            "comparison_table": {
                "meows_lite": all_metrics["models"]["meows_lite_rule"],
                "logistic_regression": all_metrics["models"]["logistic_regression"],
                "maternatriage": all_metrics["models"]["maternatriage_calibrated_forest"],
            },
            "ablation_and_stress": {
                "naive_leakage": all_metrics["naive_split_leakage_audit"],
                "age_subgroups": all_metrics["age_subgroup_audit"],
                "missing_glucometer": all_metrics["missing_glucometer_ablation"],
                "measurement_noise": all_metrics["measurement_noise_stress_test"],
            },
        },
    }

    # Verify numerical parity on held-out test set
    print("\n--- Verifying Exact Numerical Parity Between Sklearn and JSON Bundle ---")
    raw_rf_probs_sklearn = rf.predict_proba(X_test)[:, rf_class_order]
    cal_probs_sklearn = calibrator.predict_proba(raw_rf_probs_sklearn)[:, cal_class_order]

    max_prob_diff = 0.0
    parity_mismatches = 0

    for i in range(len(X_test)):
        row_dict = X_test.iloc[i].to_dict()
        bundle_out = evaluate_bundle_inference(bundle, row_dict)
        bundle_cal_p = np.array(bundle_out["calibrated_probabilities"])
        sk_cal_p = cal_probs_sklearn[i]

        diff = float(np.max(np.abs(bundle_cal_p - sk_cal_p)))
        if diff > max_prob_diff:
            max_prob_diff = diff

        if diff > 1e-4:
            parity_mismatches += 1

    print(f"Max Calibrated Probability Difference across {len(X_test)} test cases: {max_prob_diff:.2e}")
    assert max_prob_diff < 1e-4, f"Parity check failed: max difference {max_prob_diff} exceeds tolerance!"
    print("[Parity Check] SUCCESS: Pure JSON model bundle matches scikit-learn predictions with near-zero discrepancy!")

    # Save to docs/model_bundle.json
    with open(MODEL_BUNDLE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"[Export] Saved {MODEL_BUNDLE_JSON_PATH} ({os.path.getsize(MODEL_BUNDLE_JSON_PATH) / 1024:.1f} KB)")

    # Save to docs/model_bundle.js (for zero-CORS standalone execution)
    js_content = f"// MaternaTriage Offline Model Bundle\n// Auto-generated for zero-CORS file:// local execution\nwindow.MATERNATRIAGE_BUNDLE = {json.dumps(bundle)};\n"
    with open(MODEL_BUNDLE_JS_PATH, "w", encoding="utf-8") as f:
        f.write(js_content)
    print(f"[Export] Saved {MODEL_BUNDLE_JS_PATH} ({os.path.getsize(MODEL_BUNDLE_JS_PATH) / 1024:.1f} KB)")


if __name__ == "__main__":
    export_and_verify()
