import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# 1. File Path (Windows)
# =========================
DATA_PATH = r"C:\Users\thiny\OneDrive\Desktop\gittest\AI_2026\Day05 Decision tree and Essemble method\groupB\tyh_movieLens"

ratings_path = os.path.join(DATA_PATH, 'ratings.csv')
movies_path  = os.path.join(DATA_PATH, 'movies.csv')

# =========================
# 2. Load CSV Files
# =========================
ratings = pd.read_csv(ratings_path)
movies  = pd.read_csv(movies_path)

print("Ratings shape:", ratings.shape)
print("Movies shape :", movies.shape)

# =========================
# 3. Merge Ratings + Movies
# =========================
df = ratings.merge(movies, on='movieId')

# =========================
# 4. One-Hot Encode Genres
# =========================
genres = df['genres'].str.get_dummies(sep='|')

# =========================
# 5. User Statistics
# =========================
user_stats = df.groupby('userId').agg({
    'rating': ['mean', 'count', 'std']
}).reset_index()

user_stats.columns = ['userId', 'user_avg_rating', 'user_rating_count', 'user_rating_std']
user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)

# =========================
# 6. Movie Statistics
# =========================
movie_stats = df.groupby('movieId').agg({
    'rating': ['mean', 'count']
}).reset_index()

movie_stats.columns = ['movieId', 'movie_avg_rating', 'movie_rating_count']

# =========================
# 7. Final Dataset
# =========================
df_final = df.merge(user_stats, on='userId').merge(movie_stats, on='movieId')
df_final = pd.concat([df_final, genres], axis=1)

# =========================
# 8. Target Label
# =========================
df_final['high_rating'] = (df_final['rating'] >= 4).astype(int)

# =========================
# 9. Features & Target
# =========================
feature_cols = ['user_avg_rating', 'user_rating_count', 'user_rating_std',
                'movie_avg_rating', 'movie_rating_count'] + genres.columns.tolist()

X = df_final[feature_cols]
y = df_final['high_rating']

# =========================
# 10. Train / Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 11. Decision Tree (PRUNED like Image)
# =========================
dt_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,          
    min_samples_split=50,
    min_samples_leaf=20, 
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print(f"\nDecision Tree Accuracy: {dt_accuracy:.4f}")
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# =========================
# 12. Plot Tree (Readable like Image)
# =========================
plt.figure(figsize=(16, 8))  
plot_tree(
    dt_model,
    feature_names=feature_cols,
    class_names=['Low', 'High'],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3             
)
plt.title('Decision Tree (Pruned & Readable - MovieLens)')
plt.tight_layout()
plt.show()

# =========================
# 13. Random Forest (Compare)
# =========================
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# =========================
# 14. Feature Importance (Decision Tree)
# =========================

# Feature တစ်ခုချင်းစီရဲ့ အရေးပါမှုနှုန်းကို DataFrame အဖြစ် ပြုလုပ်ခြင်း
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

# ထိပ်ဆုံး Feature 10 ခုကို Bar Chart ဖြင့် ပုံဖော်ခြင်း
top_n = 10
top_features = importance.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importance (Decision Tree)')
plt.gca().invert_yaxis()   # အရေးပါမှုအများဆုံးကို အပေါ်ဆုံး
plt.tight_layout()
plt.show()

# ထိပ်ဆုံး Feature 10 ခုကို Table အဖြစ် ပြခြင်း (VS Code friendly)
print("\nTop 10 Features by Importance:")
print(top_features.to_string(index=False))
