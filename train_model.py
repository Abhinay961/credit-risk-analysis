import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score


def build_model_pipeline(X):
    numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss',
        objective='binary:logistic',
        subsample=0.8,
        colsample_bytree=0.8,
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])


def main():
    df = pd.read_csv('Loan_default.csv')
    if 'LoanID' in df.columns:
        df = df.drop(columns=['LoanID'])

    X = df.drop(columns=['Default'])
    y = df['Default']

    base_pipeline = build_model_pipeline(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        'model__n_estimators': [200, 400],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.03, 0.05],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    }

    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X, y)
    best_model = search.best_estimator_

    cv_roc_auc = cross_val_score(best_model, X, y, cv=cv, scoring='roc_auc')
    y_pred = best_model.predict(X)
    y_proba = best_model.predict_proba(X)[:, 1]

    print('Best Parameters:', search.best_params_)
    print('Mean CV ROC-AUC:', round(cv_roc_auc.mean(), 4))
    print('\nClassification Report:\n')
    print(classification_report(y, y_pred))
    print(f'Precision: {precision_score(y, y_pred):.4f}')
    print(f'Recall: {recall_score(y, y_pred):.4f}')
    print(f'ROC-AUC: {roc_auc_score(y, y_proba):.4f}')

    joblib.dump(best_model, 'credit_model.pkl')
    print('\nSaved trained model to credit_model.pkl')


if __name__ == '__main__':
    main()
