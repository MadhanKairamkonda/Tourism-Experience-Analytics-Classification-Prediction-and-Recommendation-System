import pandas as pd
import numpy as np

# =========================
# 1. LOAD EXCEL FILES
# =========================

transactions = pd.read_excel("Transaction.xlsx", engine="openpyxl")
users = pd.read_excel("User.xlsx", engine="openpyxl")
cities = pd.read_excel("City.xlsx", engine="openpyxl")
countries = pd.read_excel("Country.xlsx", engine="openpyxl")
regions = pd.read_excel("Region.xlsx", engine="openpyxl")
continents = pd.read_excel("Continent.xlsx", engine="openpyxl")
items = pd.read_excel("Item.xlsx", engine="openpyxl")
types = pd.read_excel("Type.xlsx", engine="openpyxl")
visit_modes = pd.read_excel("Mode.xlsx", engine="openpyxl")

print("Files Loaded Successfully")


# =========================
# 2. CLEAN TRANSACTION DATA
# =========================

transactions = transactions.drop_duplicates(subset="TransactionId")

transactions["Rating"] = pd.to_numeric(transactions["Rating"], errors="coerce")
transactions = transactions[(transactions["Rating"] >= 1) & (transactions["Rating"] <= 5)]

transactions["VisitYear"] = transactions["VisitYear"].astype(int)
transactions["VisitMonth"] = transactions["VisitMonth"].astype(int)

transactions["VisitDate"] = pd.to_datetime(
    transactions["VisitYear"].astype(str) + "-" +
    transactions["VisitMonth"].astype(str) + "-01",
    errors="coerce"
)

print("Transactions Cleaned")


# =========================
# 3. STANDARDIZE TEXT COLUMNS
# =========================

def standardize_text(df):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().str.lower()
    return df

cities = standardize_text(cities)
countries = standardize_text(countries)
regions = standardize_text(regions)
continents = standardize_text(continents)
items = standardize_text(items)
types = standardize_text(types)
visit_modes = standardize_text(visit_modes)

print("Text Standardized")


# =========================
# 4. MERGE ALL TABLES (SAFE HIERARCHY)
# =========================

# Merge Users
master_df = transactions.merge(users, on="UserId", how="left")

# Merge Items (Attractions)
master_df = master_df.merge(items, on="AttractionId", how="left")

# Merge Type
if "AttractionTypeId" in master_df.columns:
    master_df = master_df.merge(types, on="AttractionTypeId", how="left")

# Merge City
if "AttractionCityId" in master_df.columns:
    master_df = master_df.merge(
        cities,
        left_on="AttractionCityId",
        right_on="CityId",
        how="left"
    )

# Merge Country (ONLY if CountryId exists)
if "CountryId" in master_df.columns:
    master_df = master_df.merge(countries, on="CountryId", how="left")
else:
    print("WARNING: CountryId not found before Country merge")

# Merge Region
if "RegionId" in master_df.columns:
    master_df = master_df.merge(regions, on="RegionId", how="left")
else:
    print("WARNING: RegionId not found before Region merge")

# Merge Continent
if "ContinentId" in master_df.columns:
    master_df = master_df.merge(continents, on="ContinentId", how="left")
else:
    print("WARNING: ContinentId not found before Continent merge")

# Merge Visit Mode
if "VisitModeId" in master_df.columns:
    master_df = master_df.merge(visit_modes, on="VisitModeId", how="left")

print("All Possible Tables Merged Successfully")


# =========================
# 5. VALIDATION CHECKS
# =========================

print("\nFinal Dataset Shape:", master_df.shape)

print("\nTop Missing Columns:")
print(master_df.isnull().sum().sort_values(ascending=False).head(10))

print("\nUnique Users:", master_df["UserId"].nunique())
print("Unique Attractions:", master_df["AttractionId"].nunique())
print("Total Transactions:", len(master_df))

print("\nRating Statistics:")
print(master_df["Rating"].describe())


# =========================
# 6. SAVE CLEAN DATA
# =========================

master_df.to_csv("cleaned_master.csv", index=False)

print("\nCleaned Dataset Saved as cleaned_master.csv")