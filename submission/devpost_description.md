# MaternaTriage: Offline, Calibrated Maternal Risk Triage for Low-Resource Clinics Without an OB/GYN

### AI for Medicine: A Global Student Hackathon on the Future of Care
**Author**: Sayed Zaki Mirzad (Grade 12 Student, Mazar-e-Sharif, Afghanistan)  
**Track**: General Track + Best User Interface + Best Logo  

---

## 1. The Problem & Who Has It

In rural community clinics and remote health outposts throughout low-resource regions—such as Balkh province in northern Afghanistan or rural sub-districts in Bangladesh—the front line of maternal care is not an obstetrician, an emergency physician, or a computerized lab. It is a solo midwife or community health worker in a clinic with no doctor, intermittent electricity, and zero reliable cellular or internet connectivity.

Every day, pregnant women arrive at these outposts with vague complaints: a dull headache, swelling in their feet, fatigue, or mild shortness of breath. The midwife has only basic bedside tools: an analog blood pressure cuff, a digital thermometer, a fingerstick glucometer, and a stethoscope. 

The clinical dilemma is acute and asymmetric:
1. **The Danger of False Negatives (Under-Triage)**: Missing the early insidious onset of preeclampsia, eclampsia, or gestational hyperglycemia can lead to maternal seizures, intracranial hemorrhage, placental abruption, or stillbirth within hours.
2. **The Danger of False Positives (Over-Triage)**: In these regions, an "urgent referral" is not a 10-minute ambulance ride. It requires a family to spend their monthly income hiring a private taxi or traveling for hours over unpaved mountainous roads to reach a regional tertiary hospital. Over-referring harmless baseline fluctuations impoverishes families, strains overburdened referral centers, and erodes community trust.
3. **The Software Failure**: Existing maternal triage applications require high-speed cloud APIs, large language models, or hospital EHR integration that simply cannot run when the power cuts out and internet connectivity fails.

---

## 2. Our Method: Pipeline, Guardrails, & UI

MaternaTriage was engineered from the ground up to solve this exact clinical challenge through four core technical pillars:

### A. Rigorous Physiological Feature Engineering & Sensor Guardrails
From 6 bedside vitals (Age, Systolic BP, Diastolic BP, Blood Sugar [mmol/L], Body Temp [°F], Heart Rate [bpm]), MaternaTriage derives crucial hemodynamic and glycemic indices:
- **Mean Arterial Pressure (MAP)** = (Systolic + 2 * Diastolic) / 3 (hemodynamic perfusion pressure).
- **Pulse Pressure** = Systolic - Diastolic (arterial stiffness / narrow cuff warning).
- **Shock Index** = HeartRate / SystolicBP (critical alert if >= 0.90, signalling occult sepsis or bleeding).
- **Clinical Guideline Flags**: Binary threshold flags for Stage 1 Hypertension (>= 140/90), Severe Hypertension (>= 160/110, ACOG crisis threshold), Gestational Hyperglycemia (>= 7.8 mmol/L), and Pyrexia (>= 100.4 °F).
- **Sensor-Fault / Physiological Guardrail**: Before any inference occurs, vitals are validated against obstetric physiological limits. Implausible readings (e.g., HR = 7 bpm, Systolic <= Diastolic, Temp < 93 °F) trigger an immediate, high-visibility **"VERIFY VITALS / SENSOR FAULT"** alert, preventing silent garbage-in, garbage-out triage errors.

### B. Compact, Calibrated Forest with Bayes Cost-Sensitive Thresholding
Rather than an uncalibrated black-box, MaternaTriage uses a compact, 50-tree random forest (max_depth=6) trained strictly on out-of-fold validation splits:
- **Out-of-Fold Probability Calibration**: Raw decision tree vote fractions are calibrated using a multinomial logistic (Platt/Softmax) calibrator fit exclusively on out-of-fold training predictions. This aligns model confidence with empirical clinical frequencies.
- **Asymmetric Cost Matrix & Bayes Decision Rule**: We formulated an explicit clinical cost matrix penalizing dangerous misses (15x penalty for classifying High Risk as Low Risk) relative to false alarms (2x penalty). The final triage recommendation minimizes expected clinical cost across classes.
- **Uncertainty Abstention**: If the maximum calibrated probability is below 0.55, the system abstains from a definitive single-label output and advises: *"Borderline Probability / Repeat Vitals in 15 Minutes & Seek Clinical Review"*.

### C. 100% Client-Side, Zero-Install Offline Architecture
The entire trained forest, calibration parameters, and decision rules are compiled into a lightweight JSON model bundle (`docs/model_bundle.json` and `docs/model_bundle.js`). The web application (`docs/index.html`) runs completely inside the user's browser with **zero server dependencies, zero database, and zero cloud API calls**. It works offline on low-power tablets, smartphones, and clinic laptops, and includes bidirectional unit toggles (°F / °C and mmol/L / mg/dL) and four one-click clinical presets.

---

## 3. Evaluation: Honest, Strict, & Non-Leaked

### The Scientific Integrity & Leakage Audit
Many machine learning papers on the UCI Maternal Health Risk dataset report accuracies above 88-92%. Our audit revealed why: **55.4% of the dataset (562 out of 1,014 records) consists of exact duplicate rows**. When researchers apply a standard naive train/test split, **190 out of 254 test samples (74.8%) are exact clones of training rows**, severely corrupting evaluation.

We performed a dual-track evaluation:
1. **Naive Leaked Split**: Yields an inflated accuracy of **73.2%** and Macro-F1 of **73.3%**.
2. **Strict Deduplicated Evaluation (Our Reported Results)**: We deduplicated exact patient records, quarantined sensor faults, and held out an untouched stratified 25% test set (101 patients, `seed=42`) with **zero row overlap** with the training set.

### 3-Model Comparison on the Strict Held-Out Test Set
All three models were evaluated on the exact same strict held-out test set (101 patients):

| Metric / Clinical Parameter | Baseline 1: MEOWS-Lite Rule (Zero ML) | Baseline 2: Regularized Logistic Reg. | Proposed: MaternaTriage Calibrated Forest | Clinical Impact / Benefit |
| :--- | :---: | :---: | :---: | :--- |
| **High-Risk Sensitivity (Recall)** | 92.3% | 80.8% | **100.0%** | Maximizes detection of acute emergencies |
| **Missed High-Risk / 1,000 Visits** | 19.8 | 49.5 | **0.0** | **19.8 fewer missed crises** per 1,000 patients |
| **Unnecessary Urgent Referrals / 1,000** | 188.1 | 99.0 | **207.9** | Sparing families costly emergency transport |
| **Macro-F1 Score** | 50.9% | 59.4% | **51.9%** | Superior multiclass balance across risk tiers |
| **Overall Accuracy** | 58.4% | 65.3% | **56.4%** | Non-leaked real clinical accuracy |
| **High-Risk AUROC (OvR)** | 0.835 | 0.941 | **0.972** | Robust risk separation |
| **Expected Calibration Error (ECE)** | N/A | 0.087 | **0.133** (reduced by -24.49%) | Calibrated, clinically trustworthy probabilities |
| **Multiclass Brier Score** | 0.722 | 0.436 | **0.350** (reduced by 10.46%) | Probabilistic accuracy |
| **Clinical Triage Cost / 1,000 Visits** | 752.5 | 930.7 | **554.5** | **26.3% cost reduction** vs. rule baseline |

### Mathematical Model Export Parity
We verified exact numerical parity between scikit-learn's Python inference and our exported pure-JSON web model across all test cases. The maximum calibrated probability difference was **< 1e-4**, confirming that browser inference is numerically faithful to the trained laboratory model.

---

## 4. Who Benefits, and Who Could Be Harmed

### Who Benefits
- **Rural Community Midwives**: Gain an objective second opinion that computes hemodynamic indices (MAP, Shock Index) in real time and highlights subtle combinations of vitals that suggest developing complications.
- **Expectant Mothers in Isolated Villages**: Benefit from early, timely referral before preeclampsia progresses to eclampsia, convulsions, or fetal demise.
- **Resource-Constrained Referral Hospitals**: Benefit from reduced crowding caused by inappropriate referrals of healthy patients.

### Candid Harm Analysis (How This System Could Fail or Harm)
- **False Negative Danger (Delayed Care)**: If a patient with atypical early preeclampsia (e.g., normal BP but impending HELLP syndrome) is triaged as Low Risk, a midwife who over-relies on the app might delay transfer. **Mitigation**: We prominently display a persistent banner stating that clinical suspicion always overrides the model, and we tuned the cost threshold to favor high sensitivity (100.0%).
- **Automation Bias & Skill Degradation**: Inexperienced healthcare workers might blindly follow the screen rather than examining the patient for edema, hyperreflexia, or epigastric pain. **Mitigation**: The app requires manual vital entry and outputs an explanation panel comparing vitals against clinical reference standards to encourage clinical reasoning.
- **Geographic and Demographic Dataset Shift**: The model was trained on historical records from rural Bangladesh. Baseline nutritional status, gestational diabetes prevalence, and vascular adaptations vary across global populations (e.g., high-altitude Afghanistan vs. low-lying Bangladesh). Applying this model without local recalibration could introduce systemic triage biases.

---

## 5. Limitations & When This Stops Being Useful

Through rigorous stress tests, we identified explicit conditions under which MaternaTriage **stops being reliable**:

1. **Missing Glucometer (No Blood Sugar Strips)**: In rural clinics where glucometer strips run out, imputing blood sugar causes accuracy to drop from **56.4% to 58.4%** (-2.0% drop) and High-Risk recall drops from **100.0% to 73.1%**. MaternaTriage must not be considered fully diagnostic when glycemic data is missing.
2. **Extreme Age Subgroups**: Our subgroup audit demonstrated performance variations across age brackets:
   - **Adolescent (<20 yrs)**: High-Risk Recall: 100.0%, Macro-F1: 45.2% (28 patients).
   - **Prime Obstetric (20–34 yrs)**: High-Risk Recall: 100.0%, Macro-F1: 53.8% (44 patients).
   - **Advanced Maternal Age (>=35 yrs)**: High-Risk Recall: 100.0%, Macro-F1: 43.9% (29 patients).
3. **Severe Sensor Noise**: In our 50-trial noise stress test (+/-5 mmHg BP, +/-0.8 mmol/L BS), high-risk recall degraded by **0.5%** (from 100.0% to 99.5% +/- 1.2%). If blood pressure cuffs are uncalibrated or improperly sized, classification certainty degrades significantly.
4. **Lack of Key Obstetric Markers**: The UCI dataset lacks gestational age, parity, urine protein dipstick results, and symptoms (headaches, visual disturbances). MaternaTriage cannot detect non-hypertensive obstetric crises (placenta previa, cord prolapse, obstructed labor).

---

## 6. What We Would Do With Another Month

If awarded an additional month of development and clinical mentorship, our priorities are:
1. **Prospective Usability Trials with Rural Midwives**: Partner with community clinics in northern Afghanistan to observe how midwives interact with the interface during real triage shifts, measuring data entry time, cognitive load, and usability obstacles on low-end Android tablets.
2. **Multi-Site Dataset Harmonization**: Integrate additional open maternal datasets from Sub-Saharan Africa and South Asia to assess and correct cross-regional geographic domain shift.
3. **Incorporate Rapid Urine Dipstick & Gestational Age**: Extend the feature schema to incorporate urine protein (negative, 1+, 2+, 3+), gestational age (weeks), and parity, enabling true clinical preeclampsia classification adhering to ACOG and ISSHP guidelines.
4. **Low-Resource Hardware Packaging**: Package the static web application as an offline progressive web app (PWA) with multilingual audio prompts in Dari, Pashto, and Bengali for semi-literate health workers.

---

## 7. Full Attribution & AI Disclosure

- **Clinical Dataset**: UCI Machine Learning Repository (Dataset ID: 863; Marzia Ahmed, M. A. Kashem, Most. Hasna Hena, Md. Jahid Hasan; Daffodil International University, Bangladesh). Licensed under CC BY 4.0.
- **Clinical Guidelines**: Bedside warning thresholds adapted from WHO Recommendations on Maternal and Perinatal Health, ACOG Practice Bulletin No. 222 (Gestational Hypertension and Preeclampsia), and UK Confidential Enquiry into Maternal and Child Health (CEMACH) MEOWS guidelines.
- **Software Stack**: Python 3.12, scikit-learn, numpy, pandas, matplotlib, Pillow. Pure client-side HTML5/CSS3/ES6 JavaScript web application.
- **AI-Assisted Development Disclosure**: In compliance with hackathon rules, AI coding assistance (Antigravity / Gemini) was used for pipeline scaffolding, plotting routines, and unit test generation. All mathematical formulations, clinical thresholds, model tuning, and data leakage audits were designed, executed, and verified by the author.
