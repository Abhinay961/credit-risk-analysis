import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import sqlite3
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Industrial Credit Risk System", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f4f7f5; }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8e2dc;
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #52665b; }
    .stButton > button[kind="primary"] {
        background: #176b52;
        border: 1px solid #176b52;
    }
    .stButton > button[kind="primary"]:hover {
        background: #10543f;
        border-color: #10543f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("credit_model.pkl")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Loan_default.csv")
if "LoanID" in df.columns:
    df = df.drop(columns=["LoanID"])

feature_cols = df.drop(columns=["Default"]).columns


def compute_business_metrics(model_obj, X_eval, y_eval, threshold=0.5):
    proba = model_obj.predict_proba(X_eval)[:, 1]
    pred = (proba >= threshold).astype(int)

    precision = precision_score(y_eval, pred, zero_division=0)
    recall = recall_score(y_eval, pred, zero_division=0)
    roc_auc = roc_auc_score(y_eval, proba)
    f1 = f1_score(y_eval, pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_eval, pred).ravel()
    false_positive_cost = fp * 5000
    false_negative_cost = fn * 15000
    total_business_cost = false_positive_cost + false_negative_cost

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "business_cost": total_business_cost,
        "false_positives": fp,
        "false_negatives": fn,
    }


def risk_band_from_probability(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    return "High Risk"


def decision_from_probability(prob):
    if prob > 0.7:
        return "Reject Loan"
    elif prob > 0.4:
        return "Manual Review"
    return "Approve Loan"


def source_feature_name(encoded_name):
    feature_name = encoded_name.split("__", 1)[-1]
    for column in sorted(feature_cols, key=len, reverse=True):
        if feature_name == column or feature_name.startswith(f"{column}_"):
            return column
    return feature_name


def plain_feature_name(column):
    labels = {
        "Age": "Age",
        "Income": "Income",
        "LoanAmount": "Requested loan amount",
        "CreditScore": "Credit score",
        "MonthsEmployed": "Time in current job",
        "NumCreditLines": "Number of credit accounts",
        "InterestRate": "Loan interest rate",
        "LoanTerm": "Loan length",
        "DTIRatio": "Debt compared with income",
        "Education": "Education",
        "EmploymentType": "Employment type",
        "MaritalStatus": "Marital status",
        "HasMortgage": "Mortgage",
        "HasDependents": "Dependents",
        "LoanPurpose": "Reason for loan",
        "HasCoSigner": "Co-signer",
    }
    return labels.get(column, column.replace("_", " ").strip().capitalize())


def readable_encoded_feature_name(encoded_name):
    column = source_feature_name(encoded_name)
    label = plain_feature_name(column)
    if encoded_name.startswith("cat__"):
        encoded_level = encoded_name.split("__", 1)[-1][len(column) + 1:]
        return f"{label}: {encoded_level}"
    return label


def applicant_feature_label(column, values):
    label = plain_feature_name(column)
    if df[column].dtype == "object":
        return f"{label}: {values[column]}"
    return label

# ---------------- DATABASE ----------------
conn = sqlite3.connect("credit_history.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    probability REAL,
    cibil INTEGER,
    decision TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ---------------- UTILS ----------------
def calculate_cibil(prob):
    score = int(900 - (prob * 600))
    return max(300, min(score, 900))

def decision_from_score(score):
    if score >= 750:
        return "LOW RISK – APPROVED"
    elif score >= 650:
        return "MEDIUM RISK – HIGH INTEREST"
    elif score >= 550:
        return "HIGH RISK – LOW LIMIT"
    else:
        return "VERY HIGH RISK – REJECTED"

# ---------------- PDF ----------------
def generate_pdf(name, prob, cibil, decision):
    file_name = f"{name}_credit_report.pdf"
    pdf = SimpleDocTemplate(file_name, pagesize=A4)
    styles = getSampleStyleSheet()

    content = [
        Image("bank_logo.png", width=120, height=60),
        Paragraph("<b>CREDIT RISK ASSESSMENT REPORT</b>", styles["Title"]),
        Paragraph(f"Applicant Name: {name}", styles["Normal"]),
        Paragraph(f"Default Probability: {round(prob, 3)}", styles["Normal"]),
        Paragraph(f"CIBIL Score: {cibil}", styles["Normal"]),
        Paragraph(f"Final Decision: <b>{decision}</b>", styles["Normal"]),
        Paragraph("Generated by Industrial Credit Risk System", styles["Italic"])
    ]

    pdf.build(content)
    return file_name

# ---------------- UI ----------------
st.title("🏦 Industrial Credit Risk Assessment System")
st.caption("Credit decision support | Default risk, policy outcome, and model explanations")

name = st.text_input("Applicant Name")

input_data = {}

st.markdown("---")
st.subheader("👤 Personal Details")

for col in feature_cols:
    if col.lower() == "age":
        input_data[col] = st.number_input(
            "Age (Years)",
            min_value=18,
            max_value=100,
            step=1,
            value=30
        )

st.markdown("---")
st.subheader("💰 Loan Details")

for col in feature_cols:
    col_l = col.lower()
    if "loan" in col_l and "amount" in col_l:
        input_data[col] = st.number_input(
            "Loan Amount (in Lakhs)",
            min_value=0.0,
            step=0.1,
            value=5.0
        )
    elif "interest" in col_l:
        input_data[col] = st.number_input(
            "Interest Rate (% p.a.)",
            min_value=0.0,
            step=0.1,
            value=10.0
        )
    elif "tenure" in col_l:
        input_data[col] = st.number_input(
            "Loan Tenure (Years)",
            min_value=1,
            max_value=40,
            step=1,
            value=5
        )

st.markdown("---")
st.subheader("🧾 Credit & Employment Details")

for col in feature_cols:
    if col not in input_data:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(
                col.replace("_", " "),
                sorted(df[col].dropna().unique())
            )
        else:
            input_data[col] = st.number_input(
                col.replace("_", " "),
                step=1,
                value=0
            )

# ---------------- ASSESS ----------------
if st.button("🔍 Assess Credit Risk"):

    # Create input dataframe
    input_df = pd.DataFrame([input_data])

    # 1️⃣ ML Prediction
    prob = model.predict_proba(input_df)[0][1]
    risk_band = risk_band_from_probability(prob)
    business_decision = decision_from_probability(prob)

    # 2️⃣ Convert to CIBIL
    cibil = calculate_cibil(prob)

    # 3️⃣ Read key inputs safely
    credit_score = input_data.get("CreditScore", 999)
    income = input_data.get("Income", 999)
    past_defaults = input_data.get("PastDefaults", 0)

    # 4️⃣ BANK POLICY OVERRIDES
    if credit_score < 500:
        decision = "VERY HIGH RISK – REJECTED (Low Credit Score)"
    elif income < 2:
        decision = "VERY HIGH RISK – REJECTED (Low Income)"
    elif past_defaults >= 2:
        decision = "VERY HIGH RISK – REJECTED (Multiple Past Defaults)"
    else:
        decision = decision_from_score(cibil)

    # 5️⃣ BUSINESS DECISION SYSTEM
    if prob > 0.7:
        decision = "Reject Loan"
    elif prob > 0.4:
        decision = "Manual Review"
    else:
        decision = "Approve Loan"

    # Save history
    cur.execute(
        "INSERT INTO history (name, probability, cibil, decision) VALUES (?, ?, ?, ?)",
        (name, prob, cibil, decision)
    )
    conn.commit()

    st.markdown("---")
    st.subheader("📊 Credit Assessment Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Default Probability", round(prob, 3))
    c2.metric("CIBIL Score", cibil)
    c3.metric("Risk Band", risk_band)
    st.info(f"Recommended action: **{decision}**")
    st.progress(float(prob), text=f"Estimated default risk: {prob:.1%}")

    model_eval = compute_business_metrics(model, df.drop(columns=["Default"]), df["Default"], threshold=0.5)
    st.subheader("📈 Business Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", round(model_eval["precision"], 3))
    m2.metric("Recall", round(model_eval["recall"], 3))
    m3.metric("F1-Score", round(model_eval["f1"], 3))
    m4.metric("ROC-AUC", round(model_eval["roc_auc"], 3))
    m5.metric("Business Cost", f"₹{model_eval['business_cost']:,}")

    st.markdown("---")
    st.subheader("🧠 Model Explainability (SHAP)")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    background = preprocessor.transform(
        df.drop(columns=["Default"]).sample(50, random_state=42)
    )

    def predict_fn(x):
        return classifier.predict_proba(x)[:, 1]

    X_transformed = preprocessor.transform(input_df)
    explainer = shap.Explainer(predict_fn, background)
    shap_values = explainer(X_transformed)

    feature_names = preprocessor.get_feature_names_out()
    global_sample = preprocessor.transform(
        df.drop(columns=["Default"]).sample(20, random_state=7)
    )
    global_shap_values = explainer(global_sample)
    readable_global_values = shap.Explanation(
        values=global_shap_values.values,
        base_values=global_shap_values.base_values,
        data=global_shap_values.data,
        feature_names=[readable_encoded_feature_name(name) for name in feature_names],
    )

    applicant_contributions = {column: 0.0 for column in feature_cols}
    for encoded_name, contribution in zip(feature_names, shap_values[0].values):
        source_name = source_feature_name(encoded_name)
        if source_name in applicant_contributions:
            applicant_contributions[source_name] += float(contribution)

    grouped_names = [applicant_feature_label(column, input_data) for column in feature_cols]
    grouped_values = [applicant_contributions[column] for column in feature_cols]
    grouped_explanation = shap.Explanation(
        values=np.asarray(grouped_values),
        base_values=shap_values[0].base_values,
        feature_names=grouped_names,
    )
    top_contributors = sorted(
        applicant_contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:3]

    st.caption(
        "The first chart shows patterns across past applicants. The second shows which details moved this applicant's estimate up or down."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.write("### What Usually Matters Most")
        shap.plots.beeswarm(
            readable_global_values,
            max_display=10,
            show=False,
        )
        st.pyplot(plt.gcf())
        plt.close()

    with col2:
        st.write("### Why This Estimate Was Made")
        shap.plots.waterfall(grouped_explanation, max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.close()

    st.write("### In Everyday Terms")
    st.caption(
        "Factors below show patterns the model used, not proof that any one factor caused a future missed payment."
    )
    for column, contribution in top_contributors:
        direction = "raised" if contribution > 0 else "lowered"
        st.write(
            f"- **{applicant_feature_label(column, input_data)}** {direction} the estimated chance of a missed payment by about {abs(contribution) * 100:.1f} percentage points."
        )

    st.markdown("---")
    pdf_file = generate_pdf(name, prob, cibil, decision)
    with open(pdf_file, "rb") as f:
        st.download_button("📄 Download Credit Report", f, file_name=pdf_file)

# ---------------- HISTORY ----------------
st.markdown("---")
st.subheader("🗂 Assessment History")
history = pd.read_sql("SELECT * FROM history ORDER BY timestamp DESC", conn)
st.dataframe(history)
