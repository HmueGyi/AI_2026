
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the dataset (assume the file is in the current directory or provide full path)
df = pd.read_csv('USvideos.csv')

# Preprocess boolean columns to integers
df['comments_disabled'] = df['comments_disabled'].astype(int)
df['ratings_disabled'] = df['ratings_disabled'].astype(int)
df['video_error_or_removed'] = df['video_error_or_removed'].astype(int)

# For simplicity, drop text and date columns that aren't easily usable
drop_cols = ['video_id', 'trending_date', 'title', 'channel_title', 'publish_time', 'tags', 'thumbnail_link', 'description']

# Create a binary target: high views (1 if above median, 0 otherwise)
median_views = df['views'].median()
df['high_views'] = (df['views'] > median_views).astype(int)

# Features and target
X = df.drop(drop_cols + ['views', 'high_views'], axis=1)
y = df['high_views']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('High Views Confusion Matrix')
plt.show()

# Another confusion matrix (as in reference, with different color)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('High Views Confusion Matrix')
plt.show()

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='importance', y='feature', data=importance)
plt.title('Top Factors for High Views Prediction')
plt.show()