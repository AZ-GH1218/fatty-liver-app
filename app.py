import os
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Fatty Liver Risk Check", page_icon="🩺", layout="centered")

MODELS = {
    "完整": ("models/model_完整.joblib", "🧪 Full model", "All lab values + waist used. Highest confidence."),
    "基础": ("models/model_基础.joblib", "📊 Basic model", "Waist included; no lab values. Add lab results for higher precision."),
    "极简": ("models/model_极简.joblib", "⚪ Minimal model", "No waist or labs provided. Only age/sex/BMI used — lowest precision."),
}

bundles = {}
for k, (path, label, desc) in MODELS.items():
    if os.path.exists(path):
        bundles[k] = joblib.load(path)
    else:
        st.error(f"Missing model file: {path}")
        st.stop()

st.title("🩺 Fatty Liver Risk Check")
st.caption("A quick self-assessment · Not a medical diagnosis")

with st.form("form"):
    st.subheader("1. Basic information (required)")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 18, 100, 30)
        gender = st.selectbox("Sex", ["Male", "Female"])
    with c2:
        height = st.number_input("Height (cm)", 120.0, 220.0, 170.0, 0.5)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 65.0, 0.5)

    st.subheader("2. Waist circumference (optional)")
    use_waist = st.checkbox("I know my waist circumference")
    waist = None
    if use_waist:
        waist = st.number_input("Waist (cm)", 50.0, 160.0, 80.0, 0.5)

    st.subheader("3. Blood test results (optional)")
    c3, c4 = st.columns(2)
    with c3:
        use_alt = st.checkbox("ALT (U/L)")
        alt = st.number_input("ALT", 0.0, 500.0, 25.0, 0.5, disabled=not use_alt) if use_alt else None
        use_ast = st.checkbox("AST (U/L)")
        ast = st.number_input("AST", 0.0, 500.0, 22.0, 0.5, disabled=not use_ast) if use_ast else None
    with c4:
        use_glu = st.checkbox("Fasting glucose (mg/dL)")
        glucose = st.number_input("Glucose", 40.0, 400.0, 95.0, 0.5, disabled=not use_glu) if use_glu else None
        use_ua = st.checkbox("Uric acid (mg/dL)")
        uric = st.number_input("Uric acid", 1.0, 15.0, 5.0, 0.1, disabled=not use_ua) if use_ua else None

    submitted = st.form_submit_button("Check my risk")

if submitted:
    bmi = weight / ((height / 100) ** 2)
    gender_code = 1 if gender == "Male" else 2

    labs_all = all(v is not None for v in [alt, ast, glucose, uric])
    if labs_all and use_waist:
        tier = "完整"
        row = {"ALT": alt, "AST": ast, "Glucose": glucose, "UricAcid": uric,
               "Age": age, "Gender": gender_code, "BMI": bmi, "Waist": waist}
    elif use_waist:
        tier = "基础"
        row = {"Age": age, "Gender": gender_code, "BMI": bmi, "Waist": waist}
    else:
        tier = "极简"
        row = {"Age": age, "Gender": gender_code, "BMI": bmi}

    bundle = bundles[tier]
    X = pd.DataFrame([[row[f] for f in bundle["features"]]], columns=bundle["features"])
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    score = round(prob * 100, 1)

    if score < 30:
        level, color, advice = "Low risk", "#2e7d32", "Maintain a healthy lifestyle and have an annual check-up."
    elif score < 60:
        level, color, advice = "Moderate risk", "#ef6c00", "Consider a liver function test and abdominal ultrasound."
    else:
        level, color, advice = "High risk", "#c62828", "Please see a doctor soon for formal examination."

    st.markdown("---")
    st.subheader("Result")
    
    st.markdown(
        f"<div style='padding:20px;border-radius:12px;background:{color}20;"
        f"border-left:6px solid {color};'>"
        f"<div style='font-size:14px;color:#666;'>Fatty liver risk score (0–100)</div>"
        f"<div style='font-size:48px;font-weight:700;color:{color};'>{score}</div>"
        f"<div style='font-size:22px;font-weight:600;color:{color};'>{level}</div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown(f"**Advice:** {advice}")
    st.info(f"{MODELS[tier][1]} — {MODELS[tier][2]}")
    st.caption(f"BMI = {bmi:.1f}")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** This tool is a hackathon prototype. Results are for risk reference only and do not constitute a medical diagnosis.")
