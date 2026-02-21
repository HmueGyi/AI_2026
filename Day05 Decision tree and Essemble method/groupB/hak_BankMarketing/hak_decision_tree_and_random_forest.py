#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# SMOTE for imbalance
from imblearn.over_sampling import SMOTE

# ==================== 1. Data Loading ====================
DATA_PATH = r'bank.csv'

if not os.path.exists(DATA_PATH):
    print(f"Error: File {DATA_PATH} not found!")
    exit()

# Detect separator
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    if ';' in first_line:
        sep = ';'
    elif ',' in first_line:
        sep = ','
    else:
        sep = None

print(f"Detected separator: '{sep}'")

df = pd.read_csv(DATA_PATH, sep=sep)

print("Data Loaded Successfully!")
print("Dataset shape:", df.shape)

# ==================== 2. Target Column ကိုရှာဖွေခြင်း ====================
possible_targets = ['y', 'deposit', 'target', 'Class']
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("Error: Target column not found! Available columns:", df.columns.tolist())
    exit()

print(f"\nUsing target column: '{target_col}'")

# Check class imbalance
print("\nClass Distribution:")
print(df[target_col].value_counts())

# ==================== 3. Preprocessing ====================
categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target variable ကို 0/1 အဖြစ်ပြောင်းမယ်
if df[target_col].dtype == 'object' or pd.api.types.is_string_dtype(df[target_col]):
    df[target_col] = (df[target_col].str.lower() == 'yes').astype(int)
else:
    df[target_col] = (df[target_col] == 1).astype(int)

print("\nPreprocessing completed.")

# ==================== 4. Feature Engineering (ရိုးရှင်းအောင်ထားမယ်) ====================
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Safe division for existing features
if 'pdays' in df.columns:
    df['pdays_safe'] = df['pdays'].apply(lambda x: x if x > 0 else 1)
    df['campaign_ratio'] = df['campaign'] / df['pdays_safe']
    df = df.drop('pdays_safe', axis=1)
else:
    df['campaign_ratio'] = 0

if 'previous' in df.columns:
    df['previous_contact'] = (df['previous'] > 0).astype(int)
else:
    df['previous_contact'] = 0

if 'balance' in df.columns and 'age' in df.columns:
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
else:
    df['balance_per_age'] = 0

# Month to numeric (ရိုးရှင်းအောင်)
if 'month' in df.columns:
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month_num'] = df['month'].map(month_map).fillna(0)

# Remove inf and NaN
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Features shape: {X.shape}")

# ==================== 5. Train/Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# ==================== 6. Handle Imbalance with SMOTE ====================
print("\n" + "="*50)
print("HANDLING IMBALANCE WITH SMOTE")
print("="*50)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", pd.Series(y_train).value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# ==================== 7. Decision Tree (Simplified) ====================
print("\n" + "="*50)
print("DECISION TREE")
print("="*50)

dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5]
}

dt_base = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt_base, dt_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
dt_grid.fit(X_train_res, y_train_res)

best_dt = dt_grid.best_estimator_
print("Best Parameters:", dt_grid.best_params_)

y_pred_dt = best_dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
print(f"Test Accuracy: {dt_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# ==================== 8. Random Forest (Simplified) ====================
print("\n" + "="*50)
print("RANDOM FOREST")
print("="*50)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5],
    'max_features': ['sqrt']
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
rf_grid = GridSearchCV(rf_base, rf_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train_res, y_train_res)

best_rf = rf_grid.best_estimator_
print("Best Parameters:", rf_grid.best_params_)

y_pred_rf = best_rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Test Accuracy: {rf_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# ==================== 9. XGBoost (Simplified) ====================
print("\n" + "="*50)
print("XGBOOST")
print("="*50)

xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgb_base = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
xgb_grid = GridSearchCV(xgb_base, xgb_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train_res, y_train_res)

best_xgb = xgb_grid.best_estimator_
print("Best Parameters:", xgb_grid.best_params_)

y_pred_xgb = best_xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
print(f"Test Accuracy: {xgb_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

# ==================== 10. Voting Classifier ====================
print("\n" + "="*50)
print("VOTING CLASSIFIER")
print("="*50)

voting_model = VotingClassifier(
    estimators=[('rf', best_rf), ('xgb', best_xgb), ('dt', best_dt)],
    voting='soft'
)
voting_model.fit(X_train_res, y_train_res)

y_pred_vote = voting_model.predict(X_test)
vote_acc = accuracy_score(y_test, y_pred_vote)
print(f"Test Accuracy: {vote_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_vote))

# ==================== 11. Model Comparison ====================
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'XGBoost', 'Voting'],
    'Accuracy': [dt_acc, rf_acc, xgb_acc, vote_acc]
}).sort_values('Accuracy', ascending=False)

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(results)

# ==================== 12. Feature Importance ====================
feature_importance = pd.DataFrame({
    'feature': X.columns.tolist(),
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

print("\nTop 10 Features:")
print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*50)