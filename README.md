# Credit Risk Analysis

A machine learning-driven credit assessment system designed to predict loan default risk, estimate a credit score, and support faster lending decisions with explainable AI.

## Business Problem

Banks and financial institutions need to evaluate whether an applicant is likely to default before approving a loan. A poor decision can increase credit losses, while rejecting creditworthy applicants reduces revenue and customer trust. This project combines machine learning, business rules, and model explainability to support safer and more transparent credit decisions.

## Dataset

The project uses a loan default dataset containing customer and loan-related features such as:

- age
- income
- credit score
- loan amount
- tenure
- interest rate
- payment history
- past defaults
- employment details

The target variable is the default flag, used for supervised learning.

## Approach

1. Load and preprocess the data
2. Train a classification model for default prediction
3. Estimate default probability for each applicant
4. Convert the probability into a CIBIL-like score
5. Apply business rules to produce a lending decision
6. Explain predictions using SHAP
7. Visualize the result in a Streamlit dashboard

## Decision System

The application uses a practical risk-based decision policy:

```python
risk_prob = model.predict_proba(X)[0][1]

if risk_prob > 0.7:
    decision = "Reject Loan"
elif risk_prob > 0.4:
    decision = "Manual Review"
else:
    decision = "Approve Loan"
```

This improves the project by turning a simple probability output into a business-ready decision framework.

## Risk Segmentation

The model output is also segmented into risk bands:

- Low Risk: 0.0 to 0.3
- Medium Risk: 0.3 to 0.7
- High Risk: 0.7 to 1.0

This helps financial teams interpret risk more meaningfully than a single binary outcome.

## Features

- Credit default prediction using machine learning
- Risk probability estimation
- CIBIL-like score conversion
- Business policy-based approval decisions
- SHAP feature importance and explanation plots
- Interactive Streamlit UI
- SQLite history tracking
- PDF credit report generation

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Matplotlib
- SQLite
- ReportLab
- Joblib

## Model Comparison and Quality

The project is designed to evaluate model quality beyond basic accuracy. Important metrics include:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Cost of wrong prediction

A robust model pipeline also uses cross-validation and hyperparameter tuning to reduce variance and improve generalization.

## Explainability

The project uses SHAP to provide:

- feature importance ranking
- contribution of each variable to the prediction
- individual explanation for a specific applicant

This helps translate the model from a black box into an explainable decision support tool.

## Project Structure

```text
credit-risk-analysis/
├── app.py
├── train_model.py
├── README.md
├── requirements.txt
├── .gitignore
├── Loan_default.csv
├── credit_model.pkl
├── bank_logo.png
├── cleardb.py
├── credit_history.db
└── train_model.ipynb
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Abhinay961/credit-risk-analysis.git
cd credit-risk-analysis
```

### 2. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

## Model Training

To retrain the model with XGBoost, cross-validation, and hyperparameter tuning:

```bash
python train_model.py
```

This script trains a tuned XGBoost classifier and saves the optimized model as `credit_model.pkl`.

## Results

This project is designed to provide a practical credit risk decision workflow with transparent output. The final system balances:

- predictive accuracy
- business risk logic
- explainability
- user-friendly decision support

## Business Insights

The system helps teams identify:

- applicants likely to default
- applicants needing manual review
- low-risk applicants suitable for approval

This supports better credit management and reduces unnecessary losses.

## Future Improvements

- Add more advanced feature engineering
- Improve model performance with additional ensemble models
- Add a dashboard for executive reporting
- Connect real credit bureau APIs
- Add user authentication and admin access
- Deploy to cloud hosting
- Convert to a production API backend

## Author

Abhinay Mishra

## License

This project is for portfolio and educational use.
