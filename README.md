# 🏦 Industrial Credit Risk Assessment System

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![SHAP](https://img.shields.io/badge/Explainable-AI-success?style=for-the-badge)
![SQLite](https://img.shields.io/badge/Database-SQLite-blue?style=for-the-badge&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An AI-powered **Credit Risk Assessment System** that predicts the probability of loan default using **Machine Learning**, applies **banking policy rules**, generates a **CIBIL-like credit score**, and provides **SHAP-based explainability** for transparent lending decisions.

---

# 📌 Project Overview

Financial institutions must accurately evaluate the creditworthiness of loan applicants while maintaining transparency in their decision-making process.

The **Industrial Credit Risk Assessment System** leverages **Machine Learning** to estimate the probability of loan default using applicant financial, employment, and credit information. The predicted probability is transformed into a **CIBIL-like credit score**, followed by rule-based validation using predefined banking policies to generate the final lending decision.

To improve trust and interpretability, the system integrates **SHAP (SHapley Additive Explanations)**, allowing users to understand the contribution of every feature toward the final prediction.

The project demonstrates the practical application of **Artificial Intelligence** in the financial sector for **credit scoring**, **risk assessment**, and **decision support systems**.

---

# ✨ Features

- 🤖 Machine Learning-based loan default prediction
- 📈 Probability-based credit risk assessment
- 💳 Automatic CIBIL-like score calculation
- 🏦 Rule-based banking policy validation
- 🧠 SHAP Explainable AI visualization
- 📊 Interactive Streamlit dashboard
- 🗂 SQLite assessment history storage
- 📄 Professional PDF credit report generation
- 📉 Model explainability using SHAP Waterfall plots
- ⚡ Fast and user-friendly interface

---

# 🧠 System Workflow

```text
Applicant Details
        │
        ▼
Data Preprocessing
        │
        ▼
Machine Learning Model
        │
        ▼
Default Probability
        │
        ▼
CIBIL Score Calculation
        │
        ▼
Bank Policy Validation
        │
        ▼
Final Credit Decision
        │
        ▼
SHAP Explainability
        │
        ▼
PDF Report + Database Storage
```

---

# 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Machine Learning | Scikit-learn |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Explainable AI | SHAP |
| Visualization | Matplotlib |
| Database | SQLite |
| Report Generation | ReportLab |
| Model Serialization | Joblib |

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/abhinay/credit-risk-analysis.git

cd credit-risk-analysis
```

---

## 2️⃣ Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# 📊 Model Explainability

The system uses **SHAP (SHapley Additive Explanations)** to interpret every prediction.

For each applicant, SHAP identifies:

- Features increasing default probability
- Features reducing default probability
- Overall contribution of every feature
- Transparent reasoning behind the final prediction

This improves the trustworthiness of AI-assisted lending decisions.

---

# 📄 Credit Report Generation

After every assessment, the system automatically generates a downloadable PDF report containing:

- Applicant Name
- Default Probability
- Generated CIBIL Score
- Final Credit Decision
- Executive Summary
- System-generated Recommendation

---

# 💳 Credit Decision Logic

The final recommendation combines both:

- Machine Learning prediction
- Banking policy validation

Possible outcomes include:

- 🟢 Low Risk — Approved
- 🟡 Medium Risk — Approved with Higher Interest
- 🟠 High Risk — Low Credit Limit
- 🔴 Very High Risk — Rejected

---

# 🎯 Use Cases

- Commercial Banks
- NBFCs
- FinTech Companies
- Loan Approval Systems
- Credit Risk Assessment
- AI-powered Decision Support
- Financial Risk Analytics
- Academic Research

---

# 🚀 Future Improvements

- Real-time Banking API Integration
- Cloud Database Support
- Deep Learning Models
- User Authentication & Authorization
- Role-based Dashboard
- Docker Deployment
- Cloud Deployment (AWS / Azure / GCP)
- REST API Support
- Batch Credit Assessment

---

# 📂 Project Structure

```text
credit-risk-analysis/
│
├── app.py
├── credit_model.pkl
├── Loan_default.csv
├── requirements.txt
├── bank_logo.png
├── credit_history.db
│
└── README.md
```

---

# 👨‍💻 Author

### **Abhinay Mishra**

Full Stack Developer | Machine Learning Enthusiast | AI & Web3 Learner

🔗 GitHub:
https://github.com/abhinay

Repository:
https://github.com/abhinay/credit-risk-analysis

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub. It helps others discover the project and supports future development.