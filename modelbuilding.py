import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)



# =========================================
# 1️⃣ LOAD DATA
# =========================================
df = pd.read_csv("cleaned_master.csv")
df = df.dropna()

print("Original Shape:", df.shape)

# =========================================
# 2️⃣ SPLIT FIRST (PREVENT LEAKAGE)
# =========================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# =========================================
# 3️⃣ LEAKAGE-FREE FEATURE ENGINEERING
# =========================================

# USER FEATURES (train only)
user_avg_train = train_df.groupby("UserId")["Rating"].mean()
user_count_train = train_df.groupby("UserId")["Rating"].count()

train_df["User_Avg_Rating"] = train_df["UserId"].map(user_avg_train)
train_df["User_Total_Visits"] = train_df["UserId"].map(user_count_train)

test_df["User_Avg_Rating"] = test_df["UserId"].map(user_avg_train)
test_df["User_Total_Visits"] = test_df["UserId"].map(user_count_train)

# ATTRACTION FEATURES (train only)
attr_avg_train = train_df.groupby("AttractionId")["Rating"].mean()
attr_count_train = train_df.groupby("AttractionId")["Rating"].count()

train_df["Attraction_Avg_Rating"] = train_df["AttractionId"].map(attr_avg_train)
train_df["Attraction_Popularity"] = train_df["AttractionId"].map(attr_count_train)

test_df["Attraction_Avg_Rating"] = test_df["AttractionId"].map(attr_avg_train)
test_df["Attraction_Popularity"] = test_df["AttractionId"].map(attr_count_train)

# Handle unseen users/items in test
global_mean = train_df["Rating"].mean()

test_df.fillna({
    "User_Avg_Rating": global_mean,
    "User_Total_Visits": 1,
    "Attraction_Avg_Rating": global_mean,
    "Attraction_Popularity": 1
}, inplace=True)

print("Feature Engineering Done (Leakage-Free)")

# =========================================
# 4️⃣ DROP HIGH CARDINALITY TEXT
# =========================================
drop_cols = ["AttractionName", "City", "Country", "Region", "Continent"]
train_df = train_df.drop(columns=drop_cols, errors="ignore")
test_df = test_df.drop(columns=drop_cols, errors="ignore")

# =========================================
# 5️⃣ ENCODING
# =========================================
combined = pd.concat([train_df, test_df], axis=0)
combined_encoded = pd.get_dummies(combined, drop_first=True)

train_encoded = combined_encoded.iloc[:len(train_df)]
test_encoded = combined_encoded.iloc[len(train_df):]

# =========================================
# 6️⃣ REGRESSION — Predict Rating
# =========================================
X_train_r = train_encoded.drop(columns=["Rating"], errors="ignore")
y_train_r = train_encoded["Rating"]

X_test_r = test_encoded.drop(columns=["Rating"], errors="ignore")
y_test_r = test_encoded["Rating"]

reg = RandomForestRegressor(random_state=42)

reg_params = {
    "n_estimators": [200],
    "max_depth": [15, 20],
    "min_samples_split": [5],
    "min_samples_leaf": [2]
}

reg_grid = GridSearchCV(reg, reg_params, cv=3, n_jobs=-1)
reg_grid.fit(X_train_r, y_train_r)

best_reg = reg_grid.best_estimator_
reg_pred = best_reg.predict(X_test_r)

print("\n========== REGRESSION RESULTS ==========")
print("Best Params:", reg_grid.best_params_)
print("R2 Score:", r2_score(y_test_r, reg_pred))
print("MSE:", mean_squared_error(y_test_r, reg_pred))
print("MAE:", mean_absolute_error(y_test_r, reg_pred))

# =========================================
# 7️⃣ CLASSIFICATION — Predict VisitMode
# =========================================
if "VisitMode" in df.columns:

    y_train_c = train_encoded["VisitMode"]
    X_train_c = train_encoded.drop(columns=["VisitMode", "Rating"], errors="ignore")

    y_test_c = test_encoded["VisitMode"]
    X_test_c = test_encoded.drop(columns=["VisitMode", "Rating"], errors="ignore")

    clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    clf_params = {
        "n_estimators": [300],
        "max_depth": [15, 20],
        "min_samples_split": [5],
        "min_samples_leaf": [2]
    }

    clf_grid = GridSearchCV(clf, clf_params, cv=3, n_jobs=-1)
    clf_grid.fit(X_train_c, y_train_c)

    best_clf = clf_grid.best_estimator_
    clf_pred = best_clf.predict(X_test_c)

    print("\n========== CLASSIFICATION RESULTS ==========")
    print("Best Params:", clf_grid.best_params_)
    print("Accuracy:", accuracy_score(y_test_c, clf_pred))
    print("\nClassification Report:\n", classification_report(y_test_c, clf_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_c, clf_pred))

else:
    print("VisitMode column missing — classification skipped.")
# =========================================
# 8️⃣ COLLABORATIVE FILTERING — MATRIX FACTORIZATION (NUMPY)
# =========================================
print("\n========== COLLABORATIVE FILTERING (Matrix Factorization) ==========")

# Use SAME train/test split (no new split!)
train_cf = train_df[["UserId", "AttractionId", "Rating"]]
test_cf = test_df[["UserId", "AttractionId", "Rating"]]

# Create ID mappings
user_ids = train_cf["UserId"].unique()
item_ids = train_cf["AttractionId"].unique()

user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {a: i for i, a in enumerate(item_ids)}

num_users = len(user_ids)
num_items = len(item_ids)
k = 50  # latent factors

# Initialize latent matrices
np.random.seed(42)
P = np.random.normal(scale=0.1, size=(num_users, k))
Q = np.random.normal(scale=0.1, size=(num_items, k))

learning_rate = 0.01
reg = 0.02
epochs = 20

# Training using SGD with numeric safeguards to prevent NaNs/Infs
for epoch in range(epochs):
    for row in train_cf.itertuples(index=False):
        user, item, rating = row

        if user in user_map and item in item_map:
            u = user_map[user]
            i = item_map[item]

            # use copies so updates don't create unintended dependencies
            Pu = P[u].copy()
            Qi = Q[i].copy()

            pred = np.dot(Pu, Qi)

            # if prediction is not finite, reinitialize the vectors to small values
            if not np.isfinite(pred):
                Pu = np.random.normal(scale=0.01, size=(k,))
                Qi = np.random.normal(scale=0.01, size=(k,))
                P[u] = Pu
                Q[i] = Qi
                pred = np.dot(Pu, Qi)

            error = rating - pred

            # if error is not finite, skip this update
            if not np.isfinite(error):
                continue

            # gradient updates use the copied (pre-update) vectors
            P[u] += learning_rate * (error * Qi - reg * Pu)
            Q[i] += learning_rate * (error * Pu - reg * Qi)

    # after each epoch, sanitize matrices: replace NaN/Inf with finite numbers
    P = np.nan_to_num(P, nan=1e-8, posinf=1e8, neginf=-1e8)
    Q = np.nan_to_num(Q, nan=1e-8, posinf=1e8, neginf=-1e8)

    print(f"Epoch {epoch+1}/{epochs} completed")

# =========================================
# Evaluation on Test Set
# =========================================
preds = []
actuals = []

for row in test_cf.itertuples(index=False):
    user, item, rating = row

    if user in user_map and item in item_map:
        u = user_map[user]
        i = item_map[item]
        pred = np.dot(P[u], Q[i])
        if not np.isfinite(pred):
            pred = train_cf["Rating"].mean()
    else:
        pred = train_cf["Rating"].mean()  # cold start fallback

    preds.append(pred)
    actuals.append(rating)

# convert to arrays and filter any remaining non-finite values
preds = np.array(preds, dtype=float)
actuals = np.array(actuals, dtype=float)
mask = np.isfinite(preds) & np.isfinite(actuals)

if mask.sum() == 0:
    print("No finite predictions available for evaluation.")
else:
    rmse = np.sqrt(mean_squared_error(actuals[mask], preds[mask]))
    mae = mean_absolute_error(actuals[mask], preds[mask])

    print("Matrix Factorization RMSE:", rmse)
    print("Matrix Factorization MAE:", mae)

# =========================================
# 9️⃣ TOP-N RECOMMENDATION FUNCTION
# =========================================
def get_top_n_recommendations(user_id, n=5):
    
    if user_id not in user_map:
        # Cold start → recommend popular items
        popular_items = (
            train_cf.groupby("AttractionId")["Rating"]
            .count()
            .sort_values(ascending=False)
        )
        return list(popular_items.head(n).index)
    
    u = user_map[user_id]
    scores = []
    
    for item_id in item_ids:
        i = item_map[item_id]
        score = np.dot(P[u], Q[i])
        scores.append((item_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:n]

# Example
sample_user = train_cf["UserId"].iloc[0]
top_recs = get_top_n_recommendations(sample_user, n=5)

print(f"\nTop 5 Recommendations for User {sample_user}:")
for item, score in top_recs:
    print(f"AttractionId: {item} | Predicted Rating: {score:.2f}")

print("\nFULL PIPELINE COMPLETED SUCCESSFULLY.")

# =========================================
# 🔟 SAVE MODEL FOR STREAMLIT
# =========================================
import pickle

# Save encoded feature column order for content model
X_columns = X_train_r.columns

model_data = {
    "best_reg": best_reg,
    "P": P,
    "Q": Q,
    "user_map": user_map,
    "item_map": item_map,
    "X_columns": X_columns
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

with open("data.pkl", "wb") as f:
    pickle.dump(train_df, f)

print("\nModel and data saved successfully.")