# MaternaTriage: 2.5-Minute Video Walkthrough Script

**Speaker**: Sayed Zaki Mirzad (Grade 12, Mazar-e-Sharif, Afghanistan)  
**Total Duration**: 2 minutes, 30 seconds  
**Crucial Requirement**: Live working application must be on screen and operating within the **first 60 seconds**.

---

### [0:00 - 0:15] Phase 1: The Problem Hook (The Clinic with No Doctor)
- **Visual**: Camera on Sayed holding an analog blood pressure cuff, then cut to map of rural Afghanistan / South Asia.
- **Spoken Audio**:
  > "In rural community clinics across northern Afghanistan and South Asia, pregnant mothers are triaged by midwives with no on-site obstetrician, no lab equipment, and zero internet. When gestational hypertension or preeclampsia strikes, delayed triage can be fatal. But false alarms force impoverished families to spend a month's income on emergency transport over mountain roads. I built **MaternaTriage**—a zero-install, 100% offline clinical triage engine designed for the clinic with no doctor."

---

### [0:15 - 0:55] Phase 2: Live Working Web Application (Under 60 Seconds!)
- **Visual**: Screen capture of `docs/index.html` running in browser.
- **Actions on Screen**:
  1. *[0:18]* Click **Preset 3: Acute Preeclampsia + Hyperglycemia**. Show vitals populate instantly (BP 165/105, BS 12.0, HR 105).
  2. *[0:25]* Point to the **Color-Coded Red Triage Card** showing "HIGH RISK", immediate urgent referral guidance, and computed physiological indices: Mean Arterial Pressure (125 mmHg), Pulse Pressure (60 mmHg), and Shock Index (0.64).
  3. *[0:35]* Demonstrate **Unit Toggle**: Click °F to °C and mmol/L to mg/dL, showing seamless live unit conversion.
  4. *[0:42]* Click **Preset 4: Faulty Sensor Reading (HR = 7 bpm)**. Point to the **Amber Guardrail Banner** immediately flashing: *"VERIFY VITALS: Heart Rate 7 bpm is outside plausible bedside limits."*
- **Spoken Audio**:
  > "Here is MaternaTriage running completely offline in the browser. Watch as I click Preset 3: an acute patient with severe hypertension and hyperglycemia. The engine immediately computes Mean Arterial Pressure, Pulse Pressure, and Shock Index, triggering a calibrated High-Risk alert with clear obstetric referral steps. We support one-click metric and imperial unit switching. And if a cuff slips or an oximeter malfunctions—like this 7 bpm sensor fault—our physiological guardrail catches it immediately rather than generating a silent wrong prediction."

---

### [0:55 - 1:40] Phase 3: Scientific Integrity & Evaluation Tab
- **Visual**: Switch to Tab 2: "Held-Out Evaluation & Limitations" in the web app, scrolling through the verified metrics table and Figure 1 & Figure 2.
- **Spoken Audio**:
  > "Many published machine learning papers on this UCI dataset report over 90% accuracy. But when we audited the data, we discovered that 55% of the dataset consists of exact duplicate rows! A standard naive split leaks 190 test cases directly from training, creating an illusion of high accuracy.
  > 
  > We implemented a strict deduplicated protocol with an untouched 25% held-out test set. As shown in our live evaluation table, MaternaTriage achieves **100.0% High-Risk Sensitivity** compared to only 92.3% for the clinical MEOWS rule baseline. That means **19.8 fewer missed maternal emergencies per 1,000 patients**, while saving families over 26.3% in unnecessary referral costs."

---

### [1:40 - 2:15] Phase 4: Honest Limitations & Stress Testing
- **Visual**: Scroll down to Figures 3 and 4 in the app (Calibration Reliability and Failure Mode Ablations).
- **Spoken Audio**:
  > "True clinical AI requires honest boundaries. We calibrated our model using out-of-fold Platt scaling, reducing calibration error by -24.49%. 
  > 
  > But we also publish where this model breaks: if a clinic runs out of blood sugar test strips, our ablation shows accuracy drops by -2.0%. Furthermore, in our 50-trial measurement noise test, sensor errors reduced recall by 0.5%. MaternaTriage is a non-diagnostic decision support aid, not a doctor."

---

### [2:15 - 2:30] Phase 5: Conclusion & Next Steps
- **Visual**: Return to main dashboard with the MaternaTriage logo.
- **Spoken Audio**:
  > "With another month, we will conduct field usability trials with midwives in rural Balkh and integrate urine protein dipsticks. MaternaTriage proves that transparent, leak-free, offline medical AI can empower health workers on the front lines of maternal care. Thank you."
