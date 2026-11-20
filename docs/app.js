/**
 * MaternaTriage — Pure Client-Side Inference Engine & Medical UI Controller
 * Zero external server required. Executes offline inside any browser.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    bundle: null,
    tempUnit: "F", // "F" or "C"
    sugarUnit: "mmol", // "mmol" or "mgdl"
    currentVitalsRaw: {
      Age: 23,
      SystolicBP: 115,
      DiastolicBP: 75,
      BS: 4.8, // stored internally as mmol/L
      BodyTemp: 98.0, // stored internally as °F
      HeartRate: 76,
    },
  };

  // Clinical Reference Thresholds
  const CLINICAL_REFS = {
    map_min: 70,
    map_max: 105,
    pp_min: 30,
    pp_max: 50,
    shock_index_alert: 0.90,
    bp_severe_sys: 160,
    bp_severe_dia: 110,
    bp_stage1_sys: 140,
    bp_stage1_dia: 90,
    bs_impaired: 7.8,
    bs_severe: 11.1,
    temp_fever: 100.4,
  };

  // DOM Elements
  const elements = {
    tabTriage: document.getElementById("tab-triage"),
    tabEvaluation: document.getElementById("tab-evaluation"),
    tabBtnTriage: document.getElementById("tabBtnTriage"),
    tabBtnEvaluation: document.getElementById("tabBtnEvaluation"),
    
    // Inputs
    inputAge: document.getElementById("inputAge"),
    inputSysBP: document.getElementById("inputSysBP"),
    inputDiaBP: document.getElementById("inputDiaBP"),
    inputBS: document.getElementById("inputBS"),
    inputTemp: document.getElementById("inputTemp"),
    inputHR: document.getElementById("inputHR"),
    sugarUnitLabel: document.getElementById("sugarUnitLabel"),
    tempUnitLabel: document.getElementById("tempUnitLabel"),
    sugarHint: document.getElementById("sugarHint"),
    tempHint: document.getElementById("tempHint"),
    
    // Toggles & Presets
    toggleTemp: document.getElementById("toggleTemp"),
    toggleSugar: document.getElementById("toggleSugar"),
    btnCalculate: document.getElementById("btnCalculate"),
    btnReset: document.getElementById("btnReset"),
    preset1: document.getElementById("preset1"),
    preset2: document.getElementById("preset2"),
    preset3: document.getElementById("preset3"),
    preset4: document.getElementById("preset4"),
    
    // Outputs
    sensorFaultCard: document.getElementById("sensorFaultCard"),
    faultReasonsList: document.getElementById("faultReasonsList"),
    riskCard: document.getElementById("riskCard"),
    riskBadge: document.getElementById("riskBadge"),
    riskTitle: document.getElementById("riskTitle"),
    riskDescription: document.getElementById("riskDescription"),
    actionText: document.getElementById("actionText"),
    
    // Probability Bars
    probLowText: document.getElementById("probLowText"),
    probMidText: document.getElementById("probMidText"),
    probHighText: document.getElementById("probHighText"),
    barLow: document.getElementById("barLow"),
    barMid: document.getElementById("barMid"),
    barHigh: document.getElementById("barHigh"),
    confidenceStatus: document.getElementById("confidenceStatus"),
    
    // Hemodynamic Indices
    valMAP: document.getElementById("valMAP"),
    valPP: document.getElementById("valPP"),
    valSI: document.getElementById("valSI"),
    
    // Explanation & MEOWS
    vitalDriversList: document.getElementById("vitalDriversList"),
    meowsPredLabel: document.getElementById("meowsPredLabel"),
    meowsScoreVal: document.getElementById("meowsScoreVal"),
    meowsFlagsContainer: document.getElementById("meowsFlagsContainer"),
    metricsTableBody: document.getElementById("metricsTableBody"),
  };

  /**
   * Initialize App: load model bundle (from window or fetch)
   */
  async function init() {
    setupTabNavigation();
    setupUnitToggles();
    setupPresets();
    setupFormListeners();

    // Load Model Bundle
    if (window.MATERNATRIAGE_BUNDLE) {
      state.bundle = window.MATERNATRIAGE_BUNDLE;
      console.log("[MaternaTriage] Loaded embedded model bundle.");
      onModelReady();
    } else {
      try {
        const resp = await fetch("model_bundle.json");
        if (resp.ok) {
          state.bundle = await resp.json();
          console.log("[MaternaTriage] Loaded fetched model_bundle.json.");
          onModelReady();
        } else {
          showBundleLoadError();
        }
      } catch (err) {
        console.warn("[MaternaTriage] Fetch failed (likely file:// CORS), waiting for bundle...", err);
        showBundleLoadError();
      }
    }
  }

  function showBundleLoadError() {
    if (!state.bundle) {
      elements.confidenceStatus.textContent = "Standby: Load model_bundle.json or host via web server.";
    }
  }

  function onModelReady() {
    populateEvaluationTable();
    runInference();
  }

  /**
   * Tab Navigation Logic
   */
  function setupTabNavigation() {
    elements.tabBtnTriage.addEventListener("click", () => switchTab("tab-triage"));
    elements.tabBtnEvaluation.addEventListener("click", () => switchTab("tab-evaluation"));
  }

  function switchTab(targetId) {
    if (targetId === "tab-triage") {
      elements.tabTriage.classList.add("active");
      elements.tabEvaluation.classList.remove("active");
      elements.tabBtnTriage.classList.add("active");
      elements.tabBtnEvaluation.classList.remove("active");
    } else {
      elements.tabTriage.classList.remove("active");
      elements.tabEvaluation.classList.add("active");
      elements.tabBtnTriage.classList.remove("active");
      elements.tabBtnEvaluation.classList.add("active");
    }
  }

  /**
   * Unit Conversion Toggles
   */
  function setupUnitToggles() {
    // Temperature Toggle (°F <-> °C)
    elements.toggleTemp.addEventListener("click", () => {
      if (state.tempUnit === "F") {
        state.tempUnit = "C";
        elements.toggleTemp.textContent = "°C";
        elements.tempUnitLabel.textContent = "(°C)";
        elements.tempHint.textContent = "Fever threshold: \u226538.0 °C (100.4 °F)";
        elements.inputTemp.value = (((parseFloat(elements.inputTemp.value) - 32) * 5) / 9).toFixed(1);
        elements.inputTemp.min = "34.0";
        elements.inputTemp.max = "42.0";
      } else {
        state.tempUnit = "F";
        elements.toggleTemp.textContent = "°F";
        elements.tempUnitLabel.textContent = "(°F)";
        elements.tempHint.textContent = "Fever threshold: \u2265100.4 °F (38.0 °C)";
        elements.inputTemp.value = ((parseFloat(elements.inputTemp.value) * 9) / 5 + 32).toFixed(1);
        elements.inputTemp.min = "93.0";
        elements.inputTemp.max = "107.0";
      }
      runInference();
    });

    // Blood Sugar Toggle (mmol/L <-> mg/dL)
    elements.toggleSugar.addEventListener("click", () => {
      if (state.sugarUnit === "mmol") {
        state.sugarUnit = "mgdl";
        elements.toggleSugar.textContent = "mg/dL";
        elements.sugarUnitLabel.textContent = "(mg/dL)";
        elements.sugarHint.textContent = "WHO gestational threshold: \u2265140 mg/dL";
        elements.inputBS.value = (parseFloat(elements.inputBS.value) * 18.0182).toFixed(0);
        elements.inputBS.min = "20";
        elements.inputBS.max = "600";
        elements.inputBS.step = "1";
      } else {
        state.sugarUnit = "mmol";
        elements.toggleSugar.textContent = "mmol/L";
        elements.sugarUnitLabel.textContent = "(mmol/L)";
        elements.sugarHint.textContent = "WHO gestational threshold: \u22657.8 mmol/L";
        elements.inputBS.value = (parseFloat(elements.inputBS.value) / 18.0182).toFixed(1);
        elements.inputBS.min = "1.0";
        elements.inputBS.max = "35.0";
        elements.inputBS.step = "0.1";
      }
      runInference();
    });
  }

  /**
   * One-Click Preset Patient Scenarios
   */
  function setupPresets() {
    elements.preset1.addEventListener("click", () => {
      applyPresetVitals(23, 115, 75, 4.8, 98.0, 76);
    });
    elements.preset2.addEventListener("click", () => {
      applyPresetVitals(28, 125, 82, 6.8, 98.6, 78);
    });
    elements.preset3.addEventListener("click", () => {
      applyPresetVitals(32, 165, 105, 12.0, 99.1, 105);
    });
    elements.preset4.addEventListener("click", () => {
      applyPresetVitals(26, 120, 80, 5.2, 98.4, 7); // HR=7 bpm sensor fault
    });
  }

  function applyPresetVitals(age, sbp, dbp, bsMmol, tempF, hr) {
    elements.inputAge.value = age;
    elements.inputSysBP.value = sbp;
    elements.inputDiaBP.value = dbp;
    elements.inputHR.value = hr;

    // Set BS according to current unit
    if (state.sugarUnit === "mgdl") {
      elements.inputBS.value = (bsMmol * 18.0182).toFixed(0);
    } else {
      elements.inputBS.value = bsMmol.toFixed(1);
    }

    // Set Temp according to current unit
    if (state.tempUnit === "C") {
      elements.inputTemp.value = (((tempF - 32) * 5) / 9).toFixed(1);
    } else {
      elements.inputTemp.value = tempF.toFixed(1);
    }

    runInference();
  }

  /**
   * Form Event Listeners
   */
  function setupFormListeners() {
    const inputs = [
      elements.inputAge,
      elements.inputSysBP,
      elements.inputDiaBP,
      elements.inputBS,
      elements.inputTemp,
      elements.inputHR,
    ];
    inputs.forEach((inp) => {
      inp.addEventListener("input", () => runInference());
    });

    elements.btnCalculate.addEventListener("click", () => runInference());
    elements.btnReset.addEventListener("click", () => {
      applyPresetVitals(23, 115, 75, 4.8, 98.0, 76);
    });
  }

  /**
   * Read and Standardize Raw Vitals
   */
  function readVitals() {
    const age = parseFloat(elements.inputAge.value) || 25;
    const sbp = parseFloat(elements.inputSysBP.value) || 120;
    const dbp = parseFloat(elements.inputDiaBP.value) || 80;
    const hr = parseFloat(elements.inputHR.value) || 75;

    let bs = parseFloat(elements.inputBS.value) || 5.0;
    if (state.sugarUnit === "mgdl") {
      bs = bs / 18.0182; // convert to mmol/L for model
    }

    let temp = parseFloat(elements.inputTemp.value) || 98.0;
    if (state.tempUnit === "C") {
      temp = (temp * 9) / 5 + 32; // convert to °F for model
    }

    return {
      Age: age,
      SystolicBP: sbp,
      DiastolicBP: dbp,
      BS: bs,
      BodyTemp: temp,
      HeartRate: hr,
    };
  }

  /**
   * Physiological Sensor-Fault Check
   */
  function checkSensorFault(vitals) {
    const bounds = (state.bundle && state.bundle.sensor_bounds) || {
      age_min: 12,
      age_max: 60,
      systolic_bp_min: 60,
      systolic_bp_max: 240,
      diastolic_bp_min: 30,
      diastolic_bp_max: 160,
      bs_min: 1.0,
      bs_max: 35.0,
      body_temp_min: 93.0,
      body_temp_max: 107.0,
      heart_rate_min: 40,
      heart_rate_max: 180,
    };

    const faults = [];
    if (vitals.Age < bounds.age_min || vitals.Age > bounds.age_max) {
      faults.push(`Maternal age (${vitals.Age} yrs) outside expected obstetric range [${bounds.age_min}–${bounds.age_max}]`);
    }
    if (vitals.SystolicBP < bounds.systolic_bp_min || vitals.SystolicBP > bounds.systolic_bp_max) {
      faults.push(`Systolic BP (${vitals.SystolicBP} mmHg) outside physiological limits [${bounds.systolic_bp_min}–${bounds.systolic_bp_max}]`);
    }
    if (vitals.DiastolicBP < bounds.diastolic_bp_min || vitals.DiastolicBP > bounds.diastolic_bp_max) {
      faults.push(`Diastolic BP (${vitals.DiastolicBP} mmHg) outside physiological limits [${bounds.diastolic_bp_min}–${bounds.diastolic_bp_max}]`);
    }
    if (vitals.DiastolicBP >= vitals.SystolicBP) {
      faults.push(`Diastolic BP (${vitals.DiastolicBP}) cannot be \u2265 Systolic BP (${vitals.SystolicBP})`);
    } else if (vitals.SystolicBP - vitals.DiastolicBP < 10) {
      faults.push(`Narrow Pulse Pressure (${vitals.SystolicBP - vitals.DiastolicBP} mmHg) suggests loose cuff or severe shock`);
    }
    if (vitals.BS < bounds.bs_min || vitals.BS > bounds.bs_max) {
      faults.push(`Blood Sugar (${vitals.BS.toFixed(1)} mmol/L) outside glucometer range [${bounds.bs_min}–${bounds.bs_max}]`);
    }
    if (vitals.BodyTemp < bounds.body_temp_min || vitals.BodyTemp > bounds.body_temp_max) {
      faults.push(`Body Temp (${vitals.BodyTemp.toFixed(1)} °F) outside survivable limits [${bounds.body_temp_min}–${bounds.body_temp_max}]`);
    }
    if (vitals.HeartRate < bounds.heart_rate_min || vitals.HeartRate > bounds.heart_rate_max) {
      faults.push(`Heart Rate (${vitals.HeartRate} bpm) outside bedside physiological bounds [${bounds.heart_rate_min}–${bounds.heart_rate_max}]`);
    }

    return {
      isFault: faults.length > 0,
      faults: faults,
    };
  }

  /**
   * Feature Engineering
   */
  function deriveFeatures(vitals) {
    const sbp = vitals.SystolicBP;
    const dbp = vitals.DiastolicBP;
    const mapVal = (sbp + 2.0 * dbp) / 3.0;
    const pp = sbp - dbp;
    const si = vitals.HeartRate / Math.max(sbp, 1.0);

    return {
      Age: vitals.Age,
      SystolicBP: sbp,
      DiastolicBP: dbp,
      BS: vitals.BS,
      BodyTemp: vitals.BodyTemp,
      HeartRate: vitals.HeartRate,
      MAP: mapVal,
      PulsePressure: pp,
      ShockIndex: si,
      HypertensionFlag: sbp >= 140.0 || dbp >= 90.0 ? 1.0 : 0.0,
      SevereHTNFlag: sbp >= 160.0 || dbp >= 110.0 ? 1.0 : 0.0,
      HyperglycemiaFlag: vitals.BS >= 7.8 ? 1.0 : 0.0,
      FeverFlag: vitals.BodyTemp >= 100.4 ? 1.0 : 0.0,
    };
  }

  /**
   * Deterministic MEOWS-Lite Clinical Rule Evaluation
   */
  function evaluateMEOWSLite(vitals, features) {
    const redFlags = [];
    const yellowFlags = [];

    // BP
    if (vitals.SystolicBP >= 160 || vitals.DiastolicBP >= 110) {
      redFlags.push(`Severe Hypertension (${vitals.SystolicBP}/${vitals.DiastolicBP} mmHg)`);
    } else if (vitals.SystolicBP >= 140 || vitals.DiastolicBP >= 90) {
      yellowFlags.push(`Stage 1 Hypertension (${vitals.SystolicBP}/${vitals.DiastolicBP} mmHg)`);
    } else if (vitals.SystolicBP < 90) {
      yellowFlags.push(`Hypotension (${vitals.SystolicBP} mmHg)`);
    }

    // BS
    if (vitals.BS >= 11.1) {
      redFlags.push(`Severe Hyperglycemia (${vitals.BS.toFixed(1)} mmol/L)`);
    } else if (vitals.BS >= 7.8) {
      yellowFlags.push(`Gestational Glycemia (${vitals.BS.toFixed(1)} mmol/L)`);
    }

    // Temp
    if (vitals.BodyTemp >= 101.5) {
      redFlags.push(`High Fever (${vitals.BodyTemp.toFixed(1)} °F)`);
    } else if (vitals.BodyTemp >= 100.4) {
      yellowFlags.push(`Pyrexia (${vitals.BodyTemp.toFixed(1)} °F)`);
    }

    // HR
    if (vitals.HeartRate >= 120) {
      redFlags.push(`Severe Tachycardia (${vitals.HeartRate} bpm)`);
    } else if (vitals.HeartRate >= 100) {
      yellowFlags.push(`Tachycardia (${vitals.HeartRate} bpm)`);
    } else if (vitals.HeartRate < 50) {
      redFlags.push(`Severe Bradycardia (${vitals.HeartRate} bpm)`);
    }

    // Shock Index
    if (features.ShockIndex >= 0.90) {
      redFlags.push(`Elevated Shock Index (${features.ShockIndex.toFixed(2)})`);
    }

    let risk = "low risk";
    let score = 0;
    if (redFlags.length >= 1 || yellowFlags.length >= 2) {
      risk = "high risk";
      score = 3 + redFlags.length * 2 + yellowFlags.length;
    } else if (yellowFlags.length === 1) {
      risk = "mid risk";
      score = 1;
    }

    return {
      risk: risk,
      score: score,
      redFlags: redFlags,
      yellowFlags: yellowFlags,
    };
  }

  /**
   * Pure JS Decision Tree Evaluator
   */
  function evaluateTree(tree, featVec) {
    let node = 0;
    const left = tree.children_left;
    const right = tree.children_right;
    const feat = tree.feature;
    const thresh = tree.threshold;
    const leaf = tree.leaf_probs;

    while (left[node] !== -1) {
      const fIdx = feat[node];
      if (featVec[fIdx] <= thresh[node]) {
        node = left[node];
      } else {
        node = right[node];
      }
    }
    return leaf[node];
  }

  /**
   * Core Inference Pipeline
   */
  function runInference() {
    const vitals = readVitals();
    const sensorCheck = checkSensorFault(vitals);

    // Update Hemodynamic Indices
    const features = deriveFeatures(vitals);
    elements.valMAP.textContent = `${features.MAP.toFixed(1)} mmHg`;
    elements.valPP.textContent = `${features.PulsePressure.toFixed(0)} mmHg`;
    elements.valSI.textContent = features.ShockIndex.toFixed(2);

    // Style Shock Index if alert
    if (features.ShockIndex >= CLINICAL_REFS.shock_index_alert) {
      elements.valSI.style.color = "#E11D48";
    } else {
      elements.valSI.style.color = "#0F172A";
    }

    // 1. Guardrail Failure
    if (sensorCheck.isFault) {
      elements.sensorFaultCard.style.display = "flex";
      elements.faultReasonsList.innerHTML = sensorCheck.faults
        .map((f) => `<li>${f}</li>`)
        .join("");

      // Set card to Sensor Fault state
      elements.riskCard.className = "card risk-card card-uncertain";
      elements.riskBadge.textContent = "SENSOR FAULT";
      elements.riskTitle.textContent = "Verify Vitals";
      elements.riskDescription.textContent = "One or more bedside readings are physiologically implausible. Software triage halted.";
      elements.actionText.textContent = "Do NOT rely on software risk score. Inspect sensor placement, replace probe, re-measure vitals manually.";
      
      elements.probLowText.textContent = "--%";
      elements.probMidText.textContent = "--%";
      elements.probHighText.textContent = "--%";
      elements.barLow.style.width = "0%";
      elements.barMid.style.width = "0%";
      elements.barHigh.style.width = "0%";
      elements.confidenceStatus.textContent = "Inference halted by physiological guardrail.";
      return;
    }

    // Normal path: Hide fault banner
    elements.sensorFaultCard.style.display = "none";

    // 2. MEOWS Baseline Evaluation
    const meowsRes = evaluateMEOWSLite(vitals, features);
    elements.meowsScoreVal.textContent = meowsRes.score;
    elements.meowsPredLabel.textContent = meowsRes.risk.toUpperCase();
    if (meowsRes.risk === "high risk") {
      elements.meowsPredLabel.className = "tag-high";
    } else if (meowsRes.risk === "mid risk") {
      elements.meowsPredLabel.className = "tag-mid";
    } else {
      elements.meowsPredLabel.className = "tag-low";
    }

    const allFlags = [...meowsRes.redFlags, ...meowsRes.yellowFlags];
    if (allFlags.length > 0) {
      elements.meowsFlagsContainer.innerHTML = `<strong>Active Triggers:</strong> ${allFlags.join("; ")}`;
    } else {
      elements.meowsFlagsContainer.innerHTML = "<em>No active WHO/ACOG red or yellow warning triggers.</em>";
    }

    // 3. MaternaTriage Forest Evaluation
    if (!state.bundle) {
      return;
    }

    const bundle = state.bundle;
    const featNames = bundle.feature_names;
    const featVec = featNames.map((name) => features[name]);

    // Average tree probabilities
    const trees = bundle.forest.trees;
    const nTrees = trees.length;
    let rawProbs = [0, 0, 0];

    for (let i = 0; i < nTrees; i++) {
      const tProbs = evaluateTree(trees[i], featVec);
      rawProbs[0] += tProbs[0];
      rawProbs[1] += tProbs[1];
      rawProbs[2] += tProbs[2];
    }
    rawProbs[0] /= nTrees;
    rawProbs[1] /= nTrees;
    rawProbs[2] /= nTrees;

    // Apply Multinomial Logistic Calibrator
    const W = bundle.calibrator.weights; // (3, 3)
    const b = bundle.calibrator.intercepts; // (3,)

    const z = [
      W[0][0] * rawProbs[0] + W[0][1] * rawProbs[1] + W[0][2] * rawProbs[2] + b[0],
      W[1][0] * rawProbs[0] + W[1][1] * rawProbs[1] + W[1][2] * rawProbs[2] + b[1],
      W[2][0] * rawProbs[0] + W[2][1] * rawProbs[1] + W[2][2] * rawProbs[2] + b[2],
    ];

    const maxZ = Math.max(z[0], z[1], z[2]);
    const expZ = [Math.exp(z[0] - maxZ), Math.exp(z[1] - maxZ), Math.exp(z[2] - maxZ)];
    const sumExp = expZ[0] + expZ[1] + expZ[2];
    const calProbs = [expZ[0] / sumExp, expZ[1] / sumExp, expZ[2] / sumExp];

    // Expected Clinical Cost
    const costTable = bundle.cost_matrix_table; // (3, 3)
    const expCosts = [
      costTable[0][0] * calProbs[0] + costTable[0][1] * calProbs[1] + costTable[0][2] * calProbs[2],
      costTable[1][0] * calProbs[0] + costTable[1][1] * calProbs[1] + costTable[1][2] * calProbs[2],
      costTable[2][0] * calProbs[0] + costTable[2][1] * calProbs[1] + costTable[2][2] * calProbs[2],
    ];

    let minCostIdx = 0;
    let minCost = expCosts[0];
    for (let c = 1; c < 3; c++) {
      if (expCosts[c] < minCost) {
        minCost = expCosts[c];
        minCostIdx = c;
      }
    }

    const predictedClass = bundle.target_classes[minCostIdx];
    const maxProb = Math.max(calProbs[0], calProbs[1], calProbs[2]);
    const isUncertain = maxProb < (bundle.uncertainty_threshold || 0.55);

    // Update Probabilities UI
    const pLowPct = (calProbs[0] * 100).toFixed(1);
    const pMidPct = (calProbs[1] * 100).toFixed(1);
    const pHighPct = (calProbs[2] * 100).toFixed(1);

    elements.probLowText.textContent = `${pLowPct}%`;
    elements.probMidText.textContent = `${pMidPct}%`;
    elements.probHighText.textContent = `${pHighPct}%`;
    elements.barLow.style.width = `${pLowPct}%`;
    elements.barMid.style.width = `${pMidPct}%`;
    elements.barHigh.style.width = `${pHighPct}%`;

    // Update Risk Card Visuals
    if (isUncertain) {
      elements.riskCard.className = "card risk-card card-uncertain";
      elements.riskBadge.textContent = "BORDERLINE CONFIDENCE";
      elements.riskTitle.textContent = `${capitalize(predictedClass)} (Borderline)`;
      elements.riskDescription.textContent = `Calibrated top probability (${(maxProb * 100).toFixed(1)}%) is below decisive threshold (55%). Closely monitor vital trajectory.`;
      elements.actionText.textContent = "Repeat complete vitals in 15 minutes. Confirm patient has rested 10 minutes seated. Consult senior midwife.";
      elements.confidenceStatus.textContent = "Uncertainty Abstention Triggered: Repeat Vitals & Verify";
    } else if (predictedClass === "high risk") {
      elements.riskCard.className = "card risk-card card-high";
      elements.riskBadge.textContent = "HIGH CLINICAL RISK";
      elements.riskTitle.textContent = "High Risk";
      elements.riskDescription.textContent = "Urgent obstetric emergency indicators detected. High probability of preeclampsia, severe hypertension, or metabolic crisis.";
      elements.actionText.textContent = "INITIATE URGENT REFERRAL to hospital with surgical OB/GYN capacity. Secure IV access, administer baseline fluids/magnesium sulfate per local protocol, arrange immediate transport.";
      elements.confidenceStatus.textContent = `High Confidence (${(maxProb * 100).toFixed(1)}% calibrated certainty)`;
    } else if (predictedClass === "mid risk") {
      elements.riskCard.className = "card risk-card card-mid";
      elements.riskBadge.textContent = "MODERATE RISK";
      elements.riskTitle.textContent = "Mid Risk";
      elements.riskDescription.textContent = "Borderline or stage-1 vital abnormalities present. Patient requires enhanced surveillance.";
      elements.actionText.textContent = "Retain patient in clinic for 2-hour observation. Re-check BP and blood sugar. If persisting or rising, prepare planned transfer.";
      elements.confidenceStatus.textContent = `Decisive Prediction (${(maxProb * 100).toFixed(1)}% calibrated certainty)`;
    } else {
      elements.riskCard.className = "card risk-card card-low";
      elements.riskBadge.textContent = "LOW RISK";
      elements.riskTitle.textContent = "Low Risk";
      elements.riskDescription.textContent = "All bedside vitals within expected physiologic pregnancy boundaries.";
      elements.actionText.textContent = "Proceed with routine antenatal care checkup. Provide standard iron/folate supplementation. Schedule next routine ANC visit in 4 weeks.";
      elements.confidenceStatus.textContent = `High Confidence (${(maxProb * 100).toFixed(1)}% calibrated certainty)`;
    }

    // Local Explanations / Drivers
    updateDriversList(vitals, features, predictedClass);
  }

  function updateDriversList(vitals, feat, pred) {
    const drivers = [];

    if (vitals.SystolicBP >= 160 || vitals.DiastolicBP >= 110) {
      drivers.push(`<strong>Severe Hypertension (${vitals.SystolicBP}/${vitals.DiastolicBP} mmHg):</strong> Primary driver of acute obstetric risk.`);
    } else if (vitals.SystolicBP >= 140 || vitals.DiastolicBP >= 90) {
      drivers.push(`<strong>Stage 1 Hypertension (${vitals.SystolicBP}/${vitals.DiastolicBP} mmHg):</strong> Moderately elevates preeclampsia probability.`);
    } else {
      drivers.push(`Normal blood pressure (${vitals.SystolicBP}/${vitals.DiastolicBP} mmHg) substantially reduces cardiovascular risk.`);
    }

    if (vitals.BS >= 11.1) {
      drivers.push(`<strong>Severe Hyperglycemia (${vitals.BS.toFixed(1)} mmol/L):</strong> Triggers diabetic / metabolic complication alert.`);
    } else if (vitals.BS >= 7.8) {
      drivers.push(`<strong>Impaired Glycemia (${vitals.BS.toFixed(1)} mmol/L):</strong> Gestational hyperglycemia threshold exceeded.`);
    }

    if (feat.MAP >= 105) {
      drivers.push(`Mean Arterial Pressure is elevated (${feat.MAP.toFixed(1)} mmHg \u2265 105), indicating increased systemic vascular resistance.`);
    }

    if (vitals.HeartRate >= 100) {
      drivers.push(`Maternal tachycardia (${vitals.HeartRate} bpm) contributes to hemodynamic strain.`);
    }

    if (vitals.Age >= 35) {
      drivers.push(`Advanced maternal age (${vitals.Age} yrs) increases baseline obstetric vulnerability.`);
    } else if (vitals.Age < 20) {
      drivers.push(`Adolescent pregnancy (${vitals.Age} yrs) carries heightened hypertensive risk.`);
    }

    elements.vitalDriversList.innerHTML = drivers.map((d) => `<li>${d}</li>`).join("");
  }

  /**
   * Populate Evaluation Table in Tab 2 from Model Bundle
   */
  function populateEvaluationTable() {
    if (!state.bundle || !state.bundle.held_out_metrics) return;

    const comp = state.bundle.held_out_metrics.comparison_table;
    if (!comp) return;

    const meows = comp.meows_lite;
    const lr = comp.logistic_regression;
    const mat = comp.maternatriage;

    const rows = [
      {
        metric: "High-Risk Sensitivity (Recall)",
        meows: `${(meows.per_class_metrics["high risk"].sensitivity_recall * 100).toFixed(1)}%`,
        lr: `${(lr.per_class_metrics["high risk"].sensitivity_recall * 100).toFixed(1)}%`,
        mat: `<strong>${(mat.per_class_metrics["high risk"].sensitivity_recall * 100).toFixed(1)}%</strong>`,
      },
      {
        metric: "Missed High-Risk / 1,000 Patients",
        meows: `${meows.missed_high_risk_per_1000.toFixed(1)}`,
        lr: `${lr.missed_high_risk_per_1000.toFixed(1)}`,
        mat: `<strong>${mat.missed_high_risk_per_1000.toFixed(1)}</strong>`,
      },
      {
        metric: "Unnecessary Referrals / 1,000 Patients",
        meows: `${meows.unnecessary_urgent_referrals_per_1000.toFixed(1)}`,
        lr: `${lr.unnecessary_urgent_referrals_per_1000.toFixed(1)}`,
        mat: `<strong>${mat.unnecessary_urgent_referrals_per_1000.toFixed(1)}</strong>`,
      },
      {
        metric: "Macro-F1 Score",
        meows: `${(meows.macro_f1 * 100).toFixed(1)}%`,
        lr: `${(lr.macro_f1 * 100).toFixed(1)}%`,
        mat: `<strong>${(mat.macro_f1 * 100).toFixed(1)}%</strong>`,
      },
      {
        metric: "Overall Accuracy",
        meows: `${(meows.accuracy * 100).toFixed(1)}%`,
        lr: `${(lr.accuracy * 100).toFixed(1)}%`,
        mat: `<strong>${(mat.accuracy * 100).toFixed(1)}%</strong>`,
      },
      {
        metric: "High-Risk AUROC (One-vs-Rest)",
        meows: `${meows.high_risk_auroc.toFixed(3)}`,
        lr: `${lr.high_risk_auroc.toFixed(3)}`,
        mat: `<strong>${mat.high_risk_auroc.toFixed(3)}</strong>`,
      },
      {
        metric: "Expected Calibration Error (ECE)",
        meows: "N/A",
        lr: `${lr.calibration_ece.toFixed(3)}`,
        mat: `<strong>${mat.calibration_ece.toFixed(3)}</strong>`,
      },
      {
        metric: "Clinical Triage Cost / 1,000 Patients",
        meows: `${meows.clinical_cost_per_1000_patients.toFixed(1)}`,
        lr: `${lr.clinical_cost_per_1000_patients.toFixed(1)}`,
        mat: `<strong>${mat.clinical_cost_per_1000_patients.toFixed(1)}</strong>`,
      },
    ];

    elements.metricsTableBody.innerHTML = rows
      .map(
        (r) => `
        <tr>
          <td>${r.metric}</td>
          <td>${r.meows}</td>
          <td>${r.lr}</td>
          <td>${r.mat}</td>
        </tr>
      `
      )
      .join("");
  }

  function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  // Auto-boot on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
