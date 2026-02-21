import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # encode target
    le = LabelEncoder()
    df['deposit_encoded'] = le.fit_transform(df['deposit'])

    # categorical columns (exclude target)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if 'deposit' in cat_cols:
        cat_cols.remove('deposit')

    # one-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # features and target
    y = df_encoded['deposit_encoded']
    X = df_encoded.drop(['deposit', 'deposit_encoded'], axis=1)

    # scale numeric columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, le, scaler


def train_and_eval_dt(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return dt


def train_and_eval_rf(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return rf


def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                        param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print('RF best params:', grid.best_params_)
    print('RF best CV score:', grid.best_score_)
    return grid.best_estimator_


if __name__ == '__main__':
    csv_path = r"c:\Users\ThinkBook\Desktop\AI\AI_2026\Day05 Decision tree and Essemble method\groupB\hak_BankMarketing\bank.csv"

    print('Loading and preprocessing data...')
    X, y, label_encoder, scaler = load_and_preprocess(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print('\nTraining Decision Tree...')
    dt_model = train_and_eval_dt(X_train, X_test, y_train, y_test)

    print('\nTraining Random Forest (default)...')
    rf_model = train_and_eval_rf(X_train, X_test, y_train, y_test)

    print('\nTuning Random Forest (GridSearchCV)...')
    best_rf = tune_random_forest(X_train, y_train)
    print('\nEvaluating tuned Random Forest...')
    y_pred = best_rf.predict(X_test)
    print('Tuned RF Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # feature importances from best_rf
    try:
        importances = best_rf.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(20)
        print('\nTop 20 feature importances (tuned RF):')
        print(feat_imp)
    except Exception:
        pass

    # save models and preprocessing objects
    out_dt = r"c:\Users\ThinkBook\Desktop\AI\AI_2026\Day05 Decision tree and Essemble method\groupB\hak_BankMarketing\bank_dt_model.joblib"
    out_rf = r"c:\Users\ThinkBook\Desktop\AI\AI_2026\Day05 Decision tree and Essemble method\groupB\hak_BankMarketing\bank_rf_model.joblib"
    out_le = r"c:\Users\ThinkBook\Desktop\AI\AI_2026\Day05 Decision tree and Essemble method\groupB\hak_BankMarketing\label_encoder.joblib"
    out_scaler = r"c:\Users\ThinkBook\Desktop\AI\AI_2026\Day05 Decision tree and Essemble method\groupB\hak_BankMarketing\scaler.joblib"

    joblib.dump(dt_model, out_dt)
    joblib.dump(best_rf, out_rf)
    joblib.dump(label_encoder, out_le)
    joblib.dump(scaler, out_scaler)

    print(f'Files saved:\n - {out_dt}\n - {out_rf}\n - {out_le}\n - {out_scaler}')
