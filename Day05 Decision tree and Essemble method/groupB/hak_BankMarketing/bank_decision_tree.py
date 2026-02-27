import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# load dataset
csv_path = r"c:\Users\ThinkBook\Desktop\AI\AI_2026\Day05 Decision tree and Essemble method\groupB\hak_BankMarketing\bank.csv"
df = pd.read_csv(csv_path)
print(df.head())

# prepare data
le = LabelEncoder()
df['deposit_encoded'] = le.fit_transform(df['deposit'])
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('deposit')
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
y = df_encoded['deposit_encoded']
X = df_encoded.drop(['deposit','deposit_encoded'], axis=1)

num_cols = X.select_dtypes(include=['int64','float64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# train
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# save model
model_file = r"c:\Users\ThinkBook\Desktop\AI\AI_2026\Day05 Decision tree and Essemble method\groupB\hak_BankMarketing\bank_dt_model.joblib"
joblib.dump(clf, model_file)
print(f"Model saved to {model_file}")
