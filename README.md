# MaternaTriage

<p align="center">
  <img src="assets/logo.svg" alt="MaternaTriage Logo" width="220" />
</p>

<p align="center">
  <strong>Offline, Calibrated Maternal Risk Triage for Low-Resource Clinics Without an OB/GYN</strong><br>
  <em>Author: Sayed Zaki Mirzad &bull; Grade 12 &bull; Mazar-e-Sharif, Afghanistan</em><br>
  <em>Entry for "AI for Medicine: A Global Student Hackathon on the Future of Care" (Devpost 2026)</em>
</p>

---

> [!IMPORTANT]
> **Clinical Safety Notice**: MaternaTriage is a research decision-support prototype evaluated on historical regional clinical records. It is **NOT** a certified medical diagnostic device and **NOT** a replacement for clinical judgment by a qualified midwife or physician.

---

## 1. Executive Summary

In rural health posts across low- and middle-income regions—from northern Afghanistan to rural Bangladesh—community midwives frequently evaluate pregnant patients in facilities without on-site obstetricians, laboratory equipment, or dependable electrical connectivity. Delayed recognition of preeclampsia, gestational hypertension, and maternal hyperglycemia remains a leading driver of preventable maternal mortality worldwide.

**MaternaTriage** provides a zero-install, completely offline, bedside clinical decision-support engine. By combining physiological feature engineering (Mean Arterial Pressure, Shock Index, Pulse Pressure), sensor-fault guardrails, out-of-fold Platt probability calibration, and an asymmetric clinical cost-matrix, MaternaTriage maximizes sensitivity for acute maternal emergencies while eliminating data leakage common in published maternal ML benchmarks.

---

## 2. Scientific Integrity & Data Leakage Audit

The UCI Maternal Health Risk dataset (ID: 863; Marzia Ahmed et al., Bangladesh rural clinics, CC BY 4.0) contains **1,014 raw bedside records**. Our scientific audit uncovered critical data-quality flaws that artificially inflate accuracy in standard naive benchmarks:

1. **Exact Duplicate Rows**: **562 duplicate rows** (55.42% of the dataset) exist in the raw data.
2. **Naive Split Contamination**: A standard random train/test split leaks **190 out of 254 test cases** (74.8%) directly from the training set, creating an illusion of **73.2% accuracy** and **73.3% Macro-F1**.
3. **Strict Deduplicated Evaluation (Our Protocol)**: We quarantined sensor faults (e.g., HeartRate = 7 bpm), strictly deduplicated unique patient records, and held out an untouched **stratified 25% test set (101 patients)** with `seed=42`. Zero row overlap exists between train and test sets.

---

## 3. Verified Held-Out Test Evaluation

All metrics below reflect the **exact same strict deduplicated held-out test set** (101 patients, untouched during training, calibration, and threshold selection):

| Metric / Clinical Parameter | Baseline 1: MEOWS-Lite Rule (Zero ML) | Baseline 2: Regularized Logistic Reg. | Proposed: MaternaTriage Calibrated Forest | Clinical Impact / Benefit |
| :--- | :---: | :---: | :---: | :--- |
| **High-Risk Sensitivity (Recall)** | 92.3% | 80.8% | **100.0%** | Maximum detection of preeclampsia & crises |
| **Missed High-Risk / 1,000 Patients** | 19.8 | 49.5 | **0.0** | **19.8 fewer dangerous misses** per 1,000 |
| **Unnecessary Urgent Referrals / 1,000** | 188.1 | 99.0 | **207.9** | Prevents impoverishing emergency taxi trips |
| **Macro-F1 Score** | 50.9% | 59.4% | **51.9%** | Balanced multiclass discernment |
| **Overall Accuracy** | 58.4% | 65.3% | **56.4%** | Non-leaked strict evaluation |
| **High-Risk AUROC (OvR)** | 0.835 | 0.941 | **0.972** | High-risk discrimination |
| **Expected Calibration Error (ECE)** | N/A | 0.087 | **0.133** (reduced by -24.49%) | Reliable probabilities, not overconfident |
| **Multiclass Brier Score** | 0.722 | 0.436 | **0.350** (reduced by 10.46%) | Probabilistic accuracy |
| **Clinical Triage Cost / 1,000 Visits** | 752.5 | 930.7 | **554.5** | **26.3% cost reduction** |

---

## 4. Evaluation Visualizations

All 4 publication-grade 300-DPI evaluation figures are available in `results/figures/` and embedded in `docs/`:

1. `fig1_leakage_audit.png`: Quantifies duplicate contamination in UCI 863 and the naive accuracy inflation illusion.
2. `fig2_clinical_cost_and_recall.png`: Compares High-Risk Recall, Missed High-Risk cases per 1,000 visits, and Cost-Weight sweeps across all three models.
3. `fig3_calibration_reliability.png`: Multi-class reliability curve and ECE/Brier score reduction before and after Platt/Softmax calibration.
4. `fig4_failure_modes_ablation.png`: Subgroup recall by age (<20, 20–34, >=35 yrs), missing-glucometer test strip ablation, and 50-trial measurement noise stress test.

---

## 5. Offline Zero-Install Web App (`docs/`)

The interactive web application runs **100% locally in the browser with zero external server or Python backend**:
- **Offline Parity**: Pure client-side JavaScript engine executes `docs/model_bundle.json` with exact numerical parity (< 1e-4 probability difference) to scikit-learn.
- **Bedside Usability**:
  - Unit toggles for **°F / °C** and **mmol/L / mg/dL**.
  - **4 One-Click Clinical Presets**: Routine Normal ANC, Borderline Mid-Risk Monitoring, Acute Preeclampsia Emergency, and Sensor Fault (HR = 7 bpm).
  - Physiological Guardrail Alerts that catch detached cuffs or sensor errors before calculating risk.
  - Transparent local feature contribution breakdown and side-by-side MEOWS-Lite score comparison.
- **To Launch**: Simply open `docs/index.html` in any browser, or view via GitHub Pages.

---

## 6. Reproducibility Instructions

### Local Environment Setup (Windows, macOS, Linux)
```bash
# Clone repository
git clone https://github.com/zaki12323/maternatriage.git
cd maternatriage

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline in order:
python src/fetch_and_audit.py
python src/train_and_evaluate.py
python src/export_web_model.py
python src/generate_figures.py
python src/build_submission_docs.py
python src/verify_submission.py
```

---

## 7. Attribution & AI Disclosure

- **Clinical Dataset**: UCI Machine Learning Repository (Dataset ID: 863; Marzia Ahmed, M. A. Kashem, Most. Hasna Hena, Md. Jahid Hasan; Daffodil International University, Bangladesh). Licensed under CC BY 4.0.
- **Clinical Guidelines**: Bedside warning rules adapted from WHO Recommendations on Maternal and Perinatal Health, ACOG Practice Bulletin No. 222 (Gestational Hypertension and Preeclampsia), and UK Confidential Enquiry into Maternal and Child Health (CEMACH) MEOWS guidelines.
- **Software Stack**: Python 3.12, scikit-learn, numpy, pandas, matplotlib, Pillow.
- **AI-Assisted Development Disclosure**: In accordance with hackathon rules, AI coding assistance (Antigravity / Gemini) was utilized for automated pipeline scaffolding, formatting, and unit test generation. All mathematical formulations, clinical thresholds, model tuning, and data leakage audits were designed, executed, and verified by the author.
