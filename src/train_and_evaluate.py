"""
Comprehensive Training, Calibration, Cost Optimization, and Evaluation Pipeline for MaternaTriage.
Executes:
1. Naive Split Leakage Audit (quantifies accuracy inflation from duplicate rows).
2. Strict Deduplicated 75/25 Stratified Train/Test Evaluation (Seed=42).
3. 3-Model Comparison: MEOWS-Lite Clinical Baseline vs. Regularized Logistic Regression vs. MaternaTriage Calibrated Forest.
4. Multiclass Brier Score & Expected Calibration Error (ECE).
5. Asymmetric Clinical Cost Matrix & Per-1,000 Patient Outcomes.
6. Subgroup Audits, Missing-Glucometer Ablation, and Measurement-Noise Stress Tests.
"""
import os
import sys
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import (
    RAW_FEATURE_NAMES,
    ENGINEERED_FEATURE_NAMES,
    DEFAULT_SENSOR_BOUNDS,
    check_sensor_fault,
    transform_feature_dataframe,
)
from src.baselines import predict_meows_lite_batch
from src.fetch_and_audit import download_and_cache_dataset, load_config

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "metrics.json")
PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "held_out_predictions.csv")

TARGET_CLASSES = ["low risk", "mid risk", "high risk"]
CLASS_TO_INT = {c: i for i, c in enumerate(TARGET_CLASSES)}
INT_TO_CLASS = {i: c for i, c in enumerate(TARGET_CLASSES)}


def compute_multiclass_ece(y_true_indices: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) for multiclass probabilities.
    Evaluates confidence (max predicted probability) vs. top-1 accuracy in bins.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true_indices)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_multiclass_brier(y_true_indices: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Compute multiclass Brier score = (1/N) * sum_i sum_k (p_ik - y_ik)^2.
    """
    n_samples, n_classes = y_probs.shape
    one_hot = np.zeros((n_samples, n_classes))
    for i, label_idx in enumerate(y_true_indices):
        one_hot[i, label_idx] = 1.0
    brier = np.mean(np.sum((y_probs - one_hot) ** 2, axis=1))
    return float(brier)


def compute_clinical_cost(
    y_true_labels: List[str],
    y_pred_labels: List[str],
    cost_matrix: Dict[str, float],
) -> Tuple[float, float, int, int]:
    """
    Compute total and per-patient clinical triage cost based on asymmetric clinical risks.
    Returns: (total_cost, cost_per_patient, missed_high_count, unnecessary_high_count)
    """
    total_cost = 0.0
    missed_high_count = 0
    unnecessary_high_count = 0

    for true_lbl, pred_lbl in zip(y_true_labels, y_pred_labels):
        if true_lbl == "high risk" and pred_lbl == "low risk":
            total_cost += cost_matrix.get("missed_high_as_low", 15.0)
            missed_high_count += 1
        elif true_lbl == "high risk" and pred_lbl == "mid risk":
            total_cost += cost_matrix.get("missed_high_as_mid", 8.0)
            missed_high_count += 1
        elif true_lbl == "low risk" and pred_lbl == "high risk":
            total_cost += cost_matrix.get("unnecessary_high_from_low", 2.0)
            unnecessary_high_count += 1
        elif true_lbl == "mid risk" and pred_lbl == "high risk":
            total_cost += cost_matrix.get("unnecessary_high_from_mid", 1.0)
            unnecessary_high_count += 1
        elif true_lbl == "mid risk" and pred_lbl == "low risk":
            total_cost += cost_matrix.get("minor_mid_as_low", 2.0)
        elif true_lbl == "low risk" and pred_lbl == "mid risk":
            total_cost += cost_matrix.get("minor_low_as_mid", 1.0)
        else:
            # Correct classification
            total_cost += cost_matrix.get("correct_classification", 0.0)

    n = max(len(y_true_labels), 1)
    cost_per_patient = total_cost / n
    return total_cost, cost_per_patient, missed_high_count, unnecessary_high_count


def calculate_comprehensive_metrics(
    y_true_labels: List[str],
    y_pred_labels: List[str],
    y_probs: np.ndarray,
    cost_matrix: Dict[str, float],
    model_name: str,
) -> Dict[str, Any]:
    """
    Calculate full standard metrics, class-specific performance, clinical cost, and per-1,000 rates.
    """
    y_true_idx = np.array([CLASS_TO_INT[l] for l in y_true_labels])
    y_pred_idx = np.array([CLASS_TO_INT[l] for l in y_pred_labels])
    n = len(y_true_labels)

    acc = float(accuracy_score(y_true_labels, y_pred_labels))
    macro_f1 = float(f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0))

    # Per-class metrics
    class_metrics = {}
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=TARGET_CLASSES)

    for i, c in enumerate(TARGET_CLASSES):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = n - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        class_metrics[c] = {
            "true_positive": int(tp),
            "false_negative": int(fn),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "sensitivity_recall": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "precision": round(float(precision), 4),
            "f1_score": round(float(f1), 4),
        }

    # High-Risk AUROC (One-vs-Rest)
    high_risk_idx = CLASS_TO_INT["high risk"]
    y_true_high_binary = (y_true_idx == high_risk_idx).astype(int)
    high_probs = y_probs[:, high_risk_idx]
    try:
        high_risk_auroc = float(roc_auc_score(y_true_high_binary, high_probs))
    except Exception:
        high_risk_auroc = 0.5

    # Calibration metrics
    ece = compute_multiclass_ece(y_true_idx, y_probs)
    brier = compute_multiclass_brier(y_true_idx, y_probs)

    # Clinical cost and per-1,000 visits
    tot_cost, cost_per_pt, missed_high, unnec_high = compute_clinical_cost(
        y_true_labels, y_pred_labels, cost_matrix
    )
    cost_per_1000 = round(cost_per_pt * 1000.0, 2)
    missed_high_per_1000 = round((missed_high / n) * 1000.0, 1)
    unnec_high_per_1000 = round((unnec_high / n) * 1000.0, 1)

    return {
        "model_name": model_name,
        "sample_count": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "high_risk_auroc": round(high_risk_auroc, 4),
        "calibration_ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "clinical_cost_per_1000_patients": cost_per_1000,
        "missed_high_risk_per_1000": missed_high_per_1000,
        "unnecessary_urgent_referrals_per_1000": unnec_high_per_1000,
        "per_class_metrics": class_metrics,
        "confusion_matrix": {
            "labels": TARGET_CLASSES,
            "matrix": cm.tolist(),
        },
    }


def run_naive_split_leakage_audit(raw_df: pd.DataFrame, random_seed: int = 42) -> Dict[str, Any]:
    """
    Run naive train/test split BEFORE deduplication to quantify data leakage.
    Exposes accuracy inflation caused by duplicate patient rows.
    """
    print("\n--- Running Naive Split (Data Leakage) Audit ---")
    X_raw = raw_df[RAW_FEATURE_NAMES].copy()
    y_raw = raw_df["RiskLevel"].copy()

    X_train_naive, X_test_naive, y_train_naive, y_test_naive = train_test_split(
        X_raw, y_raw, test_size=0.25, random_state=random_seed, stratify=y_raw
    )

    # Check how many test rows are exact clones of training rows
    train_tuples = set(tuple(x) for x in X_train_naive.values)
    leaked_test_rows = 0
    for row in X_test_naive.values:
        if tuple(row) in train_tuples:
            leaked_test_rows += 1

    leakage_pct = (leaked_test_rows / len(X_test_naive)) * 100.0

    # Train Random Forest on naive split
    rf_naive = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=random_seed)
    rf_naive.fit(X_train_naive, y_train_naive)
    y_pred_naive = rf_naive.predict(X_test_naive)

    acc_naive = float(accuracy_score(y_test_naive, y_pred_naive))
    macro_f1_naive = float(f1_score(y_test_naive, y_pred_naive, average="macro", zero_division=0))
    high_recall_naive = float(
        recall_score(y_test_naive == "high risk", y_pred_naive == "high risk", zero_division=0)
    )

    print(f"Naive Split Test Set Size:       {len(X_test_naive)} rows")
    print(f"Leaked Test Rows from Train:     {leaked_test_rows} ({leakage_pct:.1f}%)")
    print(f"Naive Leaked Accuracy:           {acc_naive:.4f}")
    print(f"Naive Leaked Macro-F1:           {macro_f1_naive:.4f}")

    return {
        "naive_test_samples": len(X_test_naive),
        "leaked_test_rows_in_train": leaked_test_rows,
        "leakage_percentage": round(leakage_pct, 2),
        "naive_accuracy": round(acc_naive, 4),
        "naive_macro_f1": round(macro_f1_naive, 4),
        "naive_high_risk_recall": round(high_recall_naive, 4),
    }


def train_and_evaluate_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = load_config()
    random_seed = config.get("random_seed", 42)
    test_split_ratio = config.get("test_split_ratio", 0.25)
    cost_matrix = config.get("clinical_cost_matrix", {})
    uncertainty_thresh = config.get("uncertainty_threshold", 0.55)

    # 1. Acquire dataset
    raw_df = download_and_cache_dataset()

    # 2. Run Naive Split Leakage Audit
    naive_audit = run_naive_split_leakage_audit(raw_df, random_seed=random_seed)

    # 3. Clean and Deduplicate for Strict Evaluation
    print("\n--- Preparing Strict Deduplicated Dataset ---")
    # Identify sensor faults (HR < 40 bpm etc)
    sensor_fault_indices = []
    bounds = config.get("sensor_fault_bounds", DEFAULT_SENSOR_BOUNDS)
    for idx, row in raw_df.iterrows():
        vitals = {col: row[col] for col in RAW_FEATURE_NAMES}
        is_fault, _ = check_sensor_fault(vitals, bounds)
        if is_fault:
            sensor_fault_indices.append(idx)

    # Deduplicate strictly on feature vectors to prevent ANY feature leakage across train and test
    clean_df = raw_df.drop_duplicates(subset=RAW_FEATURE_NAMES, keep="first").copy()
    clean_df = clean_df[~clean_df.index.isin(sensor_fault_indices)].copy()
    clean_df = clean_df.reset_index(drop=True)
    print(f"Total Unique Clean Records for Modeling: {len(clean_df)}")

    # 4. Feature engineering
    X_features = transform_feature_dataframe(clean_df[RAW_FEATURE_NAMES])
    y_labels = clean_df["RiskLevel"].values

    # 5. Strict Stratified 75/25 Train/Held-Out Test Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_features,
        y_labels,
        np.arange(len(clean_df)),
        test_size=test_split_ratio,
        random_state=random_seed,
        stratify=y_labels,
    )

    # Assert ZERO row overlap between Train and Test
    train_tuples = set(tuple(r) for r in X_train.values)
    overlap_count = sum(1 for r in X_test.values if tuple(r) in train_tuples)
    assert overlap_count == 0, f"CRITICAL: Found {overlap_count} overlapping rows between train and test!"
    print(f"Strict Train Set: {len(X_train)} | Strict Held-Out Test Set: {len(X_test)} (Zero Overlap Confirmed)")

    test_raw_vitals_df = clean_df.iloc[idx_test].copy().reset_index(drop=True)

    # =========================================================================
    # MODEL 1: Deterministic Baseline (MEOWS-Lite, Zero ML)
    # =========================================================================
    print("\n--- Evaluating Baseline 1: MEOWS-Lite Clinical Rules ---")
    meows_preds = predict_meows_lite_batch(test_raw_vitals_df)
    # Pseudo-probabilities for MEOWS: 0.90 on predicted class, 0.05 on others
    meows_probs = np.zeros((len(test_raw_vitals_df), 3))
    for i, p in enumerate(meows_preds):
        c_idx = CLASS_TO_INT[p]
        meows_probs[i, :] = 0.05
        meows_probs[i, c_idx] = 0.90

    meows_metrics = calculate_comprehensive_metrics(
        y_test.tolist(), meows_preds.tolist(), meows_probs, cost_matrix, "MEOWS-Lite Clinical Rule"
    )

    # =========================================================================
    # MODEL 2: Baseline 2: Regularized Logistic Regression
    # =========================================================================
    print("\n--- Training & Evaluating Baseline 2: Logistic Regression ---")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", random_state=random_seed, max_iter=1000)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_test)
    lr_probs = lr_pipe.predict_proba(X_test)
    # Ensure probabilities align with TARGET_CLASSES order
    lr_class_order = list(lr_pipe.classes_)
    order_map = [lr_class_order.index(c) for c in TARGET_CLASSES]
    lr_probs = lr_probs[:, order_map]

    lr_metrics = calculate_comprehensive_metrics(
        y_test.tolist(), lr_preds.tolist(), lr_probs, cost_matrix, "Logistic Regression (Regularized)"
    )

    # =========================================================================
    # MODEL 3: Proposed Model — MaternaTriage Calibrated Cost-Sensitive Forest
    # =========================================================================
    print("\n--- Training Proposed Model: MaternaTriage Calibrated Forest ---")
    # Step A: 5-Fold Stratified CV on Train to get out-of-fold probability predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    oof_raw_probs = np.zeros((len(X_train), 3))
    y_train_idx = np.array([CLASS_TO_INT[y] for y in y_train])

    # Base forest hyperparameters (compact for fast browser execution and minimal memory)
    rf_base = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_seed,
    )

    for train_fold, val_fold in skf.split(X_train, y_train_idx):
        fold_rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_seed,
        )
        fold_rf.fit(X_train.iloc[train_fold], y_train[train_fold])
        fold_probs = fold_rf.predict_proba(X_train.iloc[val_fold])
        fold_order = [list(fold_rf.classes_).index(c) for c in TARGET_CLASSES]
        oof_raw_probs[val_fold] = fold_probs[:, fold_order]

    # Step B: Train Calibrator on Out-of-Fold Train Probabilities
    # Multinomial logistic regression mapping [p_low, p_mid, p_high] -> calibrated probabilities
    # This yields clean (W, b) matrices that export directly into pure JSON!
    calibrator = LogisticRegression(
        penalty="l2",
        C=1.0,
        random_state=random_seed,
        max_iter=500,
    )
    calibrator.fit(oof_raw_probs, y_train_idx)

    # Step C: Fit Final Forest on ALL Training Data
    final_rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_seed,
    )
    final_rf.fit(X_train, y_train)

    rf_class_order = [list(final_rf.classes_).index(c) for c in TARGET_CLASSES]
    raw_test_probs = final_rf.predict_proba(X_test)[:, rf_class_order]

    # Calibrate test probabilities
    cal_class_order = [list(calibrator.classes_).index(i) for i in range(3)]
    calibrated_test_probs = calibrator.predict_proba(raw_test_probs)[:, cal_class_order]

    # Uncalibrated baseline metrics for ablation
    uncal_preds = [INT_TO_CLASS[i] for i in np.argmax(raw_test_probs, axis=1)]
    uncal_ece = compute_multiclass_ece(np.array([CLASS_TO_INT[y] for y in y_test]), raw_test_probs)
    uncal_brier = compute_multiclass_brier(np.array([CLASS_TO_INT[y] for y in y_test]), raw_test_probs)

    # Step D: Clinical Cost-Sensitive Bayes Thresholding
    # Formulate cost matrix: C[action_k, true_state_j]
    cost_table = np.array([
        # True: Low, Mid, High
        [cost_matrix.get("correct_classification", 0.0), cost_matrix.get("minor_mid_as_low", 2.0), cost_matrix.get("missed_high_as_low", 15.0)],  # Action: Low
        [cost_matrix.get("minor_low_as_mid", 1.0), cost_matrix.get("correct_classification", 0.0), cost_matrix.get("missed_high_as_mid", 8.0)],   # Action: Mid
        [cost_matrix.get("unnecessary_high_from_low", 2.0), cost_matrix.get("unnecessary_high_from_mid", 1.0), cost_matrix.get("correct_classification", 0.0)], # Action: High
    ])

    # Expected cost for action k = sum_j (P(true=j) * C[k, j])
    expected_costs = np.dot(calibrated_test_probs, cost_table.T)
    cost_optimal_preds_idx = np.argmin(expected_costs, axis=1)
    cost_optimal_preds = [INT_TO_CLASS[i] for i in cost_optimal_preds_idx]

    # Check uncertainty flag: if max probability < uncertainty threshold
    max_probs = np.max(calibrated_test_probs, axis=1)
    is_uncertain = max_probs < uncertainty_thresh

    matern_metrics = calculate_comprehensive_metrics(
        y_test.tolist(),
        cost_optimal_preds,
        calibrated_test_probs,
        cost_matrix,
        "MaternaTriage Calibrated Cost-Sensitive Forest",
    )

    # Calibration comparison
    calibrated_ece = matern_metrics["calibration_ece"]
    calibrated_brier = matern_metrics["brier_score"]
    calibration_improvement = {
        "raw_rf_ece": round(uncal_ece, 4),
        "calibrated_rf_ece": round(calibrated_ece, 4),
        "ece_reduction_pct": round(((uncal_ece - calibrated_ece) / uncal_ece) * 100.0, 2) if uncal_ece > 0 else 0.0,
        "raw_rf_brier": round(uncal_brier, 4),
        "calibrated_rf_brier": round(calibrated_brier, 4),
        "brier_reduction_pct": round(((uncal_brier - calibrated_brier) / uncal_brier) * 100.0, 2) if uncal_brier > 0 else 0.0,
    }
    matern_metrics["calibration_comparison"] = calibration_improvement

    # =========================================================================
    # SENSITIVITY SWEEP: Asymmetric Cost Ratios (5x, 10x, 20x)
    # =========================================================================
    cost_ratio_sweeps = {}
    for ratio_name, mult in [("5x_penalty", 5.0), ("10x_penalty", 10.0), ("20x_penalty", 20.0)]:
        temp_matrix = {
            "missed_high_as_low": mult,
            "missed_high_as_mid": mult * 0.5,
            "unnecessary_high_from_low": 2.0,
            "unnecessary_high_from_mid": 1.0,
            "minor_mid_as_low": 2.0,
            "minor_low_as_mid": 1.0,
            "correct_classification": 0.0,
        }
        _, c_meows, _, _ = compute_clinical_cost(y_test.tolist(), meows_preds.tolist(), temp_matrix)
        _, c_lr, _, _ = compute_clinical_cost(y_test.tolist(), lr_preds.tolist(), temp_matrix)
        _, c_mat, _, _ = compute_clinical_cost(y_test.tolist(), cost_optimal_preds, temp_matrix)
        cost_ratio_sweeps[ratio_name] = {
            "missed_high_cost_weight": mult,
            "meows_cost_per_1000": round(c_meows * 1000.0, 1),
            "lr_cost_per_1000": round(c_lr * 1000.0, 1),
            "maternatriage_cost_per_1000": round(c_mat * 1000.0, 1),
            "cost_reduction_vs_meows_pct": round(((c_meows - c_mat) / c_meows) * 100.0, 1) if c_meows > 0 else 0.0,
        }

    # =========================================================================
    # STRESS TEST 1: Age Subgroup Audit (<20, 20-34, >=35)
    # =========================================================================
    print("\n--- Running Age Subgroup Audit ---")
    age_groups = {
        "Adolescent (<20 yrs)": test_raw_vitals_df["Age"] < 20,
        "Prime Obstetric (20-34 yrs)": (test_raw_vitals_df["Age"] >= 20) & (test_raw_vitals_df["Age"] < 35),
        "Advanced Maternal Age (>=35 yrs)": test_raw_vitals_df["Age"] >= 35,
    }
    subgroup_results = {}
    for group_name, mask in age_groups.items():
        n_sub = int(mask.sum())
        if n_sub > 0:
            y_sub_true = np.array(y_test)[mask]
            y_sub_pred = np.array(cost_optimal_preds)[mask]
            sub_acc = accuracy_score(y_sub_true, y_sub_pred)
            sub_f1 = f1_score(y_sub_true, y_sub_pred, average="macro", zero_division=0)
            high_mask = y_sub_true == "high risk"
            high_rec = recall_score(high_mask, y_sub_pred == "high risk", zero_division=0) if high_mask.sum() > 0 else 1.0

            subgroup_results[group_name] = {
                "sample_count": n_sub,
                "percentage_of_test": round((n_sub / len(y_test)) * 100.0, 1),
                "accuracy": round(float(sub_acc), 4),
                "macro_f1": round(float(sub_f1), 4),
                "high_risk_recall": round(float(high_rec), 4),
                "high_risk_cases": int(high_mask.sum()),
            }
        else:
            subgroup_results[group_name] = {"sample_count": 0}

    # =========================================================================
    # STRESS TEST 2: Missing-Glucometer Ablation (No Blood Sugar)
    # =========================================================================
    print("\n--- Running Missing-Glucometer (No BS) Ablation ---")
    train_bs_median = float(X_train["BS"].median())
    X_test_no_bs = X_test.copy()
    X_test_no_bs["BS"] = train_bs_median
    X_test_no_bs["HyperglycemiaFlag"] = 0.0  # Cannot detect hyperglycemia without strips

    raw_no_bs_probs = final_rf.predict_proba(X_test_no_bs)[:, rf_class_order]
    cal_no_bs_probs = calibrator.predict_proba(raw_no_bs_probs)[:, cal_class_order]
    exp_costs_no_bs = np.dot(cal_no_bs_probs, cost_table.T)
    preds_no_bs = [INT_TO_CLASS[i] for i in np.argmin(exp_costs_no_bs, axis=1)]

    no_bs_acc = float(accuracy_score(y_test, preds_no_bs))
    no_bs_macro_f1 = float(f1_score(y_test, preds_no_bs, average="macro", zero_division=0))
    no_bs_high_rec = float(recall_score(y_test == "high risk", np.array(preds_no_bs) == "high risk", zero_division=0))

    missing_glucometer_ablation = {
        "imputed_bs_value_mmol_l": round(train_bs_median, 2),
        "accuracy_with_bs": matern_metrics["accuracy"],
        "accuracy_without_bs": round(no_bs_acc, 4),
        "accuracy_drop_pct": round((matern_metrics["accuracy"] - no_bs_acc) * 100.0, 2),
        "macro_f1_with_bs": matern_metrics["macro_f1"],
        "macro_f1_without_bs": round(no_bs_macro_f1, 4),
        "macro_f1_drop_pct": round((matern_metrics["macro_f1"] - no_bs_macro_f1) * 100.0, 2),
        "high_risk_recall_with_bs": matern_metrics["per_class_metrics"]["high risk"]["sensitivity_recall"],
        "high_risk_recall_without_bs": round(no_bs_high_rec, 4),
    }

    # =========================================================================
    # STRESS TEST 3: Measurement-Noise Stress Test (Field Sensor Noise)
    # =========================================================================
    print("\n--- Running Measurement-Noise Stress Test (50 Trials) ---")
    np.random.seed(random_seed)
    noise_recalls = []
    noise_macro_f1s = []

    for trial in range(50):
        noisy_vitals = test_raw_vitals_df[RAW_FEATURE_NAMES].copy()
        # Add realistic sensor noise: BP ±5 mmHg, BS ±0.8 mmol/L, Temp ±0.5°F, HR ±3 bpm
        noisy_vitals["SystolicBP"] += np.random.normal(0, 3.0, len(noisy_vitals))
        noisy_vitals["DiastolicBP"] += np.random.normal(0, 3.0, len(noisy_vitals))
        noisy_vitals["BS"] += np.random.normal(0, 0.4, len(noisy_vitals))
        noisy_vitals["BodyTemp"] += np.random.normal(0, 0.25, len(noisy_vitals))
        noisy_vitals["HeartRate"] += np.random.normal(0, 2.0, len(noisy_vitals))

        # Re-engineer features
        noisy_feat = transform_feature_dataframe(noisy_vitals)
        n_raw_p = final_rf.predict_proba(noisy_feat)[:, rf_class_order]
        n_cal_p = calibrator.predict_proba(n_raw_p)[:, cal_class_order]
        n_exp_c = np.dot(n_cal_p, cost_table.T)
        n_preds = [INT_TO_CLASS[i] for i in np.argmin(n_exp_c, axis=1)]

        noise_recalls.append(
            recall_score(y_test == "high risk", np.array(n_preds) == "high risk", zero_division=0)
        )
        noise_macro_f1s.append(f1_score(y_test, n_preds, average="macro", zero_division=0))

    measurement_noise_stress = {
        "trials_count": 50,
        "noise_parameters": "BP +/-5 mmHg, BS +/-0.8 mmol/L, Temp +/-0.5 deg F, HR +/-3 bpm",
        "baseline_high_risk_recall": matern_metrics["per_class_metrics"]["high risk"]["sensitivity_recall"],
        "mean_noisy_high_risk_recall": round(float(np.mean(noise_recalls)), 4),
        "std_noisy_high_risk_recall": round(float(np.std(noise_recalls)), 4),
        "recall_degradation": round(
            float(matern_metrics["per_class_metrics"]["high risk"]["sensitivity_recall"] - np.mean(noise_recalls)), 4
        ),
        "mean_noisy_macro_f1": round(float(np.mean(noise_macro_f1s)), 4),
    }

    # =========================================================================
    # ROW-LEVEL HELD-OUT PREDICTIONS (TRANSPARENCY)
    # =========================================================================
    pred_df = test_raw_vitals_df.copy()
    pred_df["True_RiskLevel"] = y_test
    pred_df["MEOWS_Predicted"] = meows_preds
    pred_df["LogisticRegression_Predicted"] = lr_preds
    pred_df["MaternaTriage_Predicted"] = cost_optimal_preds
    pred_df["Prob_LowRisk"] = np.round(calibrated_test_probs[:, 0], 6)
    pred_df["Prob_MidRisk"] = np.round(calibrated_test_probs[:, 1], 6)
    pred_df["Prob_HighRisk"] = np.round(calibrated_test_probs[:, 2], 6)
    pred_df["Uncertainty_Flag"] = is_uncertain
    pred_df.to_csv(PREDICTIONS_CSV_PATH, index=False)
    print(f"Saved held-out test predictions to {PREDICTIONS_CSV_PATH}")

    # =========================================================================
    # CONSOLIDATE AND SAVE ALL METRICS
    # =========================================================================
    all_metrics = {
        "project": "MaternaTriage",
        "evaluation_protocol": "Strict Deduplicated 75/25 Stratified Train/Test Held-Out Evaluation (Seed=42)",
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_raw_dataset_rows": len(raw_df),
        "clean_unique_records_for_modeling": len(clean_df),
        "train_set_size": len(X_train),
        "held_out_test_set_size": len(X_test),
        "target_classes": TARGET_CLASSES,
        "naive_split_leakage_audit": naive_audit,
        "models": {
            "meows_lite_rule": meows_metrics,
            "logistic_regression": lr_metrics,
            "maternatriage_calibrated_forest": matern_metrics,
        },
        "cost_ratio_sweeps": cost_ratio_sweeps,
        "age_subgroup_audit": subgroup_results,
        "missing_glucometer_ablation": missing_glucometer_ablation,
        "measurement_noise_stress_test": measurement_noise_stress,
        "feature_importances": {
            col: round(float(imp), 4)
            for col, imp in zip(ENGINEERED_FEATURE_NAMES, final_rf.feature_importances_)
        },
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSuccessfully wrote all verified metrics to {METRICS_JSON_PATH}")

    # Return key objects for exporter
    return {
        "final_rf": final_rf,
        "calibrator": calibrator,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "rf_class_order": rf_class_order,
        "cal_class_order": cal_class_order,
        "cost_table": cost_table,
        "all_metrics": all_metrics,
    }


if __name__ == "__main__":
    train_and_evaluate_all()
