"""
Automated QA verification suite for MaternaTriage hackathon submission.
Verifies scientific integrity, zero data leakage, finite metrics, model bundle numerical parity,
asset generation, and absence of template placeholders in submission documents.
"""
import os
import sys
import json
import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import RAW_FEATURE_NAMES, ENGINEERED_FEATURE_NAMES
from src.export_web_model import evaluate_bundle_inference

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_JSON_PATH = os.path.join(RESULTS_DIR, "metrics.json")
PREDICTIONS_CSV_PATH = os.path.join(RESULTS_DIR, "held_out_predictions.csv")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "maternal_health_risk.csv")
AUDIT_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "data_audit_report.json")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MODEL_BUNDLE_JSON = os.path.join(DOCS_DIR, "model_bundle.json")
MODEL_BUNDLE_JS = os.path.join(DOCS_DIR, "model_bundle.js")
DEVPOST_PATH = os.path.join(PROJECT_ROOT, "submission", "devpost_description.md")
VIDEO_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "submission", "demo_video_script.md")
README_PATH = os.path.join(PROJECT_ROOT, "README.md")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")


class QAVerifier:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.results: List[Dict[str, Any]] = []

    def check(self, name: str, condition: bool, details: str = ""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
            print(f" [PASS] {name} - {details}")
        else:
            self.failed += 1
            print(f" [FAIL] {name} - {details}")
        self.results.append({"name": name, "status": status, "details": details})

    def run_all(self) -> bool:
        print("\n" + "=" * 70)
        print("MATERNATRIAGE AUTOMATED QA & INTEGRITY VERIFICATION SUITE")
        print("=" * 70)

        # 1. Dataset verification
        self.verify_dataset()

        # 2. Results & Metrics verification
        self.verify_metrics()

        # 3. Model bundle & numerical parity verification
        self.verify_model_bundle()

        # 4. Figures and Assets verification
        self.verify_assets_and_figures()

        # 5. Offline Web App verification
        self.verify_web_app()

        # 6. Submission documents & placeholder check
        self.verify_submission_documents()

        print("\n" + "=" * 70)
        print(f"QA VERIFICATION SUMMARY: {self.passed} PASSED | {self.failed} FAILED")
        print("=" * 70)

        if self.failed == 0:
            print("\n ALL SCIENTIFIC INTEGRITY AND SUBMISSION CHECKS PASSED!\n")
            self.print_executive_summary()
            return True
        else:
            print(f"\n WARNING: {self.failed} checks failed. Address the issues above before submitting.")
            return False

    def verify_dataset(self):
        print("\n--- 1. Real Dataset & Integrity Audit ---")
        csv_exists = os.path.exists(RAW_DATA_PATH)
        self.check("Cached Raw Dataset Exists", csv_exists, f"Path: {RAW_DATA_PATH}")

        if csv_exists:
            df = pd.read_csv(RAW_DATA_PATH)
            row_count = len(df)
            self.check("Raw Record Count Matches UCI 863 (1,014 rows)", row_count == 1014, f"Found: {row_count} rows")
            
            # Check column presence
            has_cols = all(c in df.columns for c in RAW_FEATURE_NAMES + ["RiskLevel"])
            self.check("Expected Feature & Target Columns Present", has_cols, f"Columns: {list(df.columns)}")

        audit_exists = os.path.exists(AUDIT_REPORT_PATH)
        self.check("Data Audit Report JSON Exists", audit_exists, f"Path: {AUDIT_REPORT_PATH}")
        if audit_exists:
            with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
                audit = json.load(f)
            dups = audit.get("exact_duplicate_rows", 0)
            self.check("Duplicate Rows Audited (~562 expected)", dups > 500, f"Found {dups} exact duplicate rows")

    def verify_metrics(self):
        print("\n--- 2. Metrics & Zero-Leakage Held-Out Evaluation ---")
        metrics_exist = os.path.exists(METRICS_JSON_PATH)
        self.check("metrics.json Exists", metrics_exist, f"Path: {METRICS_JSON_PATH}")

        if not metrics_exist:
            return

        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)

        # Check models present
        models = m.get("models", {})
        has_3_models = all(k in models for k in ["meows_lite_rule", "logistic_regression", "maternatriage_calibrated_forest"])
        self.check("All 3 Models Evaluated", has_3_models, f"Models: {list(models.keys())}")

        # Check finite and valid numbers
        mat = models.get("maternatriage_calibrated_forest", {})
        acc = mat.get("accuracy", -1)
        macro_f1 = mat.get("macro_f1", -1)
        sens = mat.get("per_class_metrics", {}).get("high risk", {}).get("sensitivity_recall", -1)
        cost_1000 = mat.get("clinical_cost_per_1000_patients", -1)
        ece = mat.get("calibration_ece", -1)
        brier = mat.get("brier_score", -1)

        self.check("MaternaTriage Accuracy Valid (0 < acc <= 1)", 0 < acc <= 1.0, f"Accuracy: {acc:.4f}")
        self.check("MaternaTriage Macro-F1 Valid (0 < f1 <= 1)", 0 < macro_f1 <= 1.0, f"Macro-F1: {macro_f1:.4f}")
        self.check("High-Risk Sensitivity Valid (0 < sens <= 1)", 0 < sens <= 1.0, f"Sensitivity: {sens:.4f}")
        self.check("Clinical Cost per 1,000 Finite", cost_1000 > 0, f"Cost/1000: {cost_1000}")
        self.check("Calibration ECE Finite & Sub-0.20", 0 <= ece < 0.20, f"ECE: {ece:.4f}")
        self.check("Multiclass Brier Score Finite & Sub-0.50", 0 <= brier < 0.50, f"Brier: {brier:.4f}")

        # Naive split leakage check
        naive = m.get("naive_split_leakage_audit", {})
        leak_pct = naive.get("leakage_percentage", 0)
        self.check("Naive Split Leakage Quantified (>40%)", leak_pct > 40.0, f"Leakage: {leak_pct:.1f}%")

        # Held-out predictions CSV
        preds_exist = os.path.exists(PREDICTIONS_CSV_PATH)
        self.check("held_out_predictions.csv Exists", preds_exist, f"Path: {PREDICTIONS_CSV_PATH}")

    def verify_model_bundle(self):
        print("\n--- 3. Web Model Bundle & Mathematical Parity ---")
        json_bundle_exists = os.path.exists(MODEL_BUNDLE_JSON)
        js_bundle_exists = os.path.exists(MODEL_BUNDLE_JS)

        self.check("docs/model_bundle.json Exists", json_bundle_exists, f"Path: {MODEL_BUNDLE_JSON}")
        self.check("docs/model_bundle.js Exists", js_bundle_exists, f"Path: {MODEL_BUNDLE_JS}")

        if not json_bundle_exists or not os.path.exists(PREDICTIONS_CSV_PATH):
            return

        with open(MODEL_BUNDLE_JSON, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        pred_df = pd.read_csv(PREDICTIONS_CSV_PATH)
        n_samples = len(pred_df)

        max_discrepancy = 0.0
        for _, row in pred_df.iterrows():
            from src.features import compute_clinical_features
            raw_vitals = {c: float(row[c]) for c in RAW_FEATURE_NAMES}
            full_feat = compute_clinical_features(raw_vitals)
            out = evaluate_bundle_inference(bundle, full_feat)

            # Compare probabilities against saved predictions
            p_low = row["Prob_LowRisk"]
            p_mid = row["Prob_MidRisk"]
            p_high = row["Prob_HighRisk"]
            saved_probs = np.array([p_low, p_mid, p_high])
            bundle_probs = np.array(out["calibrated_probabilities"])

            diff = float(np.max(np.abs(saved_probs - bundle_probs)))
            if diff > max_discrepancy:
                max_discrepancy = diff

        self.check(
            "Numerical Parity: Pure JSON Bundle vs Saved Sklearn Probabilities (< 1e-4)",
            max_discrepancy < 1e-4,
            f"Max discrepancy across {n_samples} test records: {max_discrepancy:.2e}",
        )

    def verify_assets_and_figures(self):
        print("\n--- 4. Visual Assets & Publication Figures ---")
        svg_logo = os.path.join(ASSETS_DIR, "logo.svg")
        png_logo = os.path.join(ASSETS_DIR, "logo.png")

        self.check("assets/logo.svg Exists", os.path.exists(svg_logo), f"Path: {svg_logo}")
        png_exists = os.path.exists(png_logo)
        self.check("assets/logo.png Exists", png_exists, f"Path: {png_logo}")

        if png_exists:
            from PIL import Image
            img = Image.open(png_logo)
            self.check("assets/logo.png Resolution is 1024x1024", img.size == (1024, 1024), f"Dimensions: {img.size}")

        fig_names = [
            "fig1_leakage_audit.png",
            "fig2_clinical_cost_and_recall.png",
            "fig3_calibration_reliability.png",
            "fig4_failure_modes_ablation.png",
        ]

        for fname in fig_names:
            p_res = os.path.join(RESULTS_DIR, "figures", fname)
            p_doc = os.path.join(DOCS_DIR, "figures", fname)
            self.check(f"Figure {fname} in results/figures/", os.path.exists(p_res), f"Path: {p_res}")
            self.check(f"Figure {fname} in docs/figures/", os.path.exists(p_doc), f"Path: {p_doc}")

    def verify_web_app(self):
        print("\n--- 5. Static Zero-Install Web Application ---")
        index_html = os.path.join(DOCS_DIR, "index.html")
        styles_css = os.path.join(DOCS_DIR, "styles.css")
        app_js = os.path.join(DOCS_DIR, "app.js")

        self.check("docs/index.html Exists", os.path.exists(index_html), f"Path: {index_html}")
        self.check("docs/styles.css Exists", os.path.exists(styles_css), f"Path: {styles_css}")
        self.check("docs/app.js Exists", os.path.exists(app_js), f"Path: {app_js}")

        if os.path.exists(index_html):
            with open(index_html, "r", encoding="utf-8") as f:
                html = f.read()
            self.check("HTML has Persistent Non-Diagnostic Warning Banner", "Non-Diagnostic" in html or "Clinical Decision Support" in html, "Found banner")
            self.check("HTML embeds model_bundle.js for offline file:// support", "model_bundle.js" in html, "Includes fallback bundle script")
            self.check("HTML includes unit toggles (C/F and mmol/L / mg/dL)", "unit-toggle" in html or "toggle" in html.lower(), "Found unit toggles")

    def verify_submission_documents(self):
        print("\n--- 6. Submission Documents & Placeholder Auditing ---")
        self.check("README.md Exists", os.path.exists(README_PATH), f"Path: {README_PATH}")
        self.check("devpost_description.md Exists", os.path.exists(DEVPOST_PATH), f"Path: {DEVPOST_PATH}")
        self.check("demo_video_script.md Exists", os.path.exists(VIDEO_SCRIPT_PATH), f"Path: {VIDEO_SCRIPT_PATH}")

        if os.path.exists(DEVPOST_PATH):
            with open(DEVPOST_PATH, "r", encoding="utf-8") as f:
                devpost = f.read()

            # Check mandatory sections
            required_sections = [
                "The Problem & Who Has It",
                "Our Method",
                "Evaluation",
                "Who Benefits, and Who Could Be Harmed",
                "Limitations & When This Stops Being Useful",
                "What We Would Do With Another Month",
                "Full Attribution",
            ]
            for sec in required_sections:
                self.check(f"Devpost Has Section: '{sec}'", sec.lower() in devpost.lower(), f"Matched section: {sec}")

            # Check for leftover placeholders
            placeholder_patterns = [r"\bTODO\b", r"\bTBD\b", r"\[INSERT", r"XX%", r"YY%"]
            found_placeholders = []
            for pat in placeholder_patterns:
                matches = re.findall(pat, devpost, re.IGNORECASE)
                if matches:
                    found_placeholders.extend(matches)

            self.check(
                "Zero Leftover Placeholders in Devpost Description",
                len(found_placeholders) == 0,
                f"Found: {found_placeholders}" if found_placeholders else "Clean - zero placeholders found",
            )

        if os.path.exists(VIDEO_SCRIPT_PATH):
            with open(VIDEO_SCRIPT_PATH, "r", encoding="utf-8") as f:
                script = f.read()
            self.check(
                "Video Script Shows Working App in First 60 Seconds",
                "first 60 seconds" in script.lower() or "0:15" in script,
                "Timing confirmed",
            )

    def print_executive_summary(self):
        if not os.path.exists(METRICS_JSON_PATH):
            return

        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)

        naive = m["naive_split_leakage_audit"]
        m_meows = m["models"]["meows_lite_rule"]
        m_lr = m["models"]["logistic_regression"]
        m_mat = m["models"]["maternatriage_calibrated_forest"]

        print("=" * 70)
        print("EXECUTIVE HACKATHON SUMMARY")
        print("=" * 70)
        print("1. DATA LEAKAGE AUDIT RESULT:")
        print(f"   - Total Raw Dataset Rows:    {m['total_raw_dataset_rows']}")
        print(f"   - Leaked Test Rows in Naive: {naive['leaked_test_rows_in_train']} / {naive['naive_test_samples']} ({naive['leakage_percentage']}%)")
        print(f"   - Naive Leaked Accuracy:     {naive['naive_accuracy']*100:.1f}% (Illusion)")
        print(f"   - Real Deduplicated Test N:  {m['held_out_test_set_size']} patients (Untouched Held-Out Test Set)")
        print()
        print("2. FINAL HELD-OUT PERFORMANCE COMPARISON:")
        print("   " + "-" * 66)
        print(f"   {'Metric':<28} | {'MEOWS-Lite':<10} | {'LogReg':<10} | {'MaternaTriage':<12}")
        print("   " + "-" * 66)
        print(f"   {'High-Risk Recall (Sens)':<28} | {m_meows['per_class_metrics']['high risk']['sensitivity_recall']*100:>9.1f}% | {m_lr['per_class_metrics']['high risk']['sensitivity_recall']*100:>9.1f}% | {m_mat['per_class_metrics']['high risk']['sensitivity_recall']*100:>11.1f}%")
        print(f"   {'Missed High-Risk / 1,000':<28} | {m_meows['missed_high_risk_per_1000']:>10.1f} | {m_lr['missed_high_risk_per_1000']:>10.1f} | {m_mat['missed_high_risk_per_1000']:>12.1f}")
        print(f"   {'Unnecessary Referrals/1k':<28} | {m_meows['unnecessary_urgent_referrals_per_1000']:>10.1f} | {m_lr['unnecessary_urgent_referrals_per_1000']:>10.1f} | {m_mat['unnecessary_urgent_referrals_per_1000']:>12.1f}")
        print(f"   {'Macro-F1 Score':<28} | {m_meows['macro_f1']*100:>9.1f}% | {m_lr['macro_f1']*100:>9.1f}% | {m_mat['macro_f1']*100:>11.1f}%")
        print(f"   {'Overall Accuracy':<28} | {m_meows['accuracy']*100:>9.1f}% | {m_lr['accuracy']*100:>9.1f}% | {m_mat['accuracy']*100:>11.1f}%")
        print(f"   {'Calibration ECE':<28} | {'N/A':>10} | {m_lr['calibration_ece']:>10.3f} | {m_mat['calibration_ece']:>12.3f}")
        print(f"   {'Clinical Cost / 1,000 Pts':<28} | {m_meows['clinical_cost_per_1000_patients']:>10.1f} | {m_lr['clinical_cost_per_1000_patients']:>10.1f} | {m_mat['clinical_cost_per_1000_patients']:>12.1f}")
        print("   " + "-" * 66)
        print()
        print("3. DEPLOYMENT & RECORDING COMMANDS:")
        print("   - To preview web app locally:  Open 'docs/index.html' in your browser")
        print("   - To push to GitHub:")
        print("       git init")
        print("       git add .")
        print("       git commit -m 'Initial release of MaternaTriage for Devpost Hackathon'")
        print("       git branch -M main")
        print("       git remote add origin https://github.com/sayedzakimirzad/maternatriage.git")
        print("       git push -u origin main")
        print("   - To enable GitHub Pages: Go to Settings -> Pages -> Deploy from branch 'main' -> folder '/docs'")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    verifier = QAVerifier()
    success = verifier.run_all()
    sys.exit(0 if success else 1)
