import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


def load_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'loan_approval_dataset.csv')
    df = pd.read_csv(path)
    return df


def preprocess(df, target_column=None):
    df = df.copy()
    # Identify target
    if target_column is None:
        # common target name
        possible = [c for c in df.columns if c.lower() in ('loan_status','status','approved','target')]
        target_column = possible[0] if possible else df.columns[-1]

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Impute numeric with median, categorical with mode
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(num_cols) > 0:
        num_imp = SimpleImputer(strategy='median')
        X[num_cols] = num_imp.fit_transform(X[num_cols])
    if len(cat_cols) > 0:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imp.fit_transform(X[cat_cols])

    # Encode categorical features
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le

    # Encode target if categorical
    target_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    return X, y, {'num_cols': num_cols, 'cat_cols': cat_cols, 'encoders': encoders, 'target_encoder': target_encoder}


def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(report)
    return {'accuracy': acc, 'report': report}


def save_model(model, path):
    joblib.dump(model, path)


if __name__ == '__main__':
    df = load_data()
    X, y, meta = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = train_decision_tree(X_train, y_train)
    print('\nDecision Tree evaluation:')
    evaluate_model(dt, X_test, y_test)
    save_model(dt, 'nlh_decision_tree.joblib')

    rf = train_random_forest(X_train, y_train)
    print('\nRandom Forest evaluation:')
    evaluate_model(rf, X_test, y_test)
    save_model(rf, 'nlh_random_forest.joblib')
