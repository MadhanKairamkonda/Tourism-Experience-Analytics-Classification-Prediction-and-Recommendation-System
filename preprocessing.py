import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# 1. LOAD CLEANED DATA
# =========================

df = pd.read_csv("cleaned_master.csv")

print("Dataset Loaded:", df.shape)

# =========================
# 2. HANDLE MISSING VALUES
# =========================

# Drop rows where critical columns are missing
critical_cols = ["UserId", "AttractionId", "Rating"]
df = df.dropna(subset=critical_cols)

# Fill categorical nulls with 'unknown'
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].fillna("unknown")

# Fill numeric nulls with median
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

print("Missing Values Handled")

# =========================
# 3. FEATURE ENGINEERING
# =========================

# User-level features
user_features = df.groupby("UserId").agg({
    "Rating": ["mean", "count"],
    "VisitYear": "nunique",
    "AttractionId": "nunique"
}).reset_index()

user_features.columns = [
    "UserId",
    "User_Avg_Rating",
    "User_Total_Visits",
    "User_Active_Years",
    "User_Unique_Attractions"
]

df = df.merge(user_features, on="UserId", how="left")

print("User-level Features Created")

# Attraction-level features
attraction_features = df.groupby("AttractionId").agg({
    "Rating": ["mean", "count"]
}).reset_index()

attraction_features.columns = [
    "AttractionId",
    "Attraction_Avg_Rating",
    "Attraction_Total_Visits"
]

df = df.merge(attraction_features, on="AttractionId", how="left")

print("Attraction-level Features Created")

# =========================
# 4. ENCODING CATEGORICAL VARIABLES
# =========================

categorical_cols = [
    "VisitMode",
    "Continent",
    "Country",
    "Region",
    "AttractionType"
]

label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print("Categorical Encoding Done")

# =========================
# 5. NORMALIZATION
# =========================

scaler = StandardScaler()

numeric_cols = [
    "Rating",
    "User_Avg_Rating",
    "User_Total_Visits",
    "User_Active_Years",
    "User_Unique_Attractions",
    "Attraction_Avg_Rating",
    "Attraction_Total_Visits"
]

existing_numeric = [col for col in numeric_cols if col in df.columns]

df[existing_numeric] = scaler.fit_transform(df[existing_numeric])

print("Normalization Completed")

# =========================
# 6. SAVE PROCESSED DATA
# =========================

df.to_csv("processed_master.csv", index=False)

print("Processed Dataset Saved:", df.shape)