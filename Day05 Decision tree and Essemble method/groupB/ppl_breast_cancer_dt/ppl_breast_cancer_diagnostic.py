import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('breast_cancer_diagnostic_data.csv')

print('Data shape:', df.shape)


df.columns = df.columns.str.strip()


df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Drop unnamed columns
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

X = df.drop(columns=['id', 'diagnosis'])
y = df['diagnosis']

print('Fixed columns:', df.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = DecisionTreeClassifier(max_depth=5, random_state=123)
model.fit(X_train, y_train)

print('Training accuracy:', model.score(X_train, y_train))

y_pred = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(5, 8))
sns.barplot(x='importance', y='feature', data=importance)
plt.title('Top Factors for Breast Cancer Diagnosis')
plt.show()