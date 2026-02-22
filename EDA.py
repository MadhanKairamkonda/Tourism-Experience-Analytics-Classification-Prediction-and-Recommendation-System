import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("processed_master.csv")

sns.set_style("whitegrid")

print("Dataset Shape:", df.shape)

# ==============================
# 1️⃣ Rating Distribution
# ==============================
plt.figure()
sns.histplot(df["Rating"], bins=30, kde=True)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()


# ==============================
# 2️⃣ Visit Mode Distribution
# ==============================
if "VisitMode" in df.columns:
    plt.figure()
    df["VisitMode"].value_counts().plot(kind="bar")
    plt.title("Visit Mode Distribution")
    plt.xlabel("Visit Mode")
    plt.ylabel("Count")
    plt.show()


# ==============================
# 3️⃣ Top 10 Popular Attractions
# ==============================
top_attractions = (
    df.groupby("AttractionId")["Attraction_Total_Visits"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure()
top_attractions.plot(kind="bar")
plt.title("Top 10 Most Popular Attractions")
plt.xlabel("AttractionId")
plt.ylabel("Total Visits")
plt.show()


# ==============================
# 4️⃣ User Activity Distribution
# ==============================
plt.figure()
sns.histplot(df["User_Total_Visits"], bins=40)
plt.title("User Total Visits Distribution")
plt.xlabel("Total Visits")
plt.ylabel("Frequency")
plt.show()


# ==============================
# 5️⃣ Correlation Heatmap
# ==============================
plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()


# ==============================
# 6️⃣ Rating vs Attraction Popularity
# ==============================
plt.figure()
sns.scatterplot(
    x="Attraction_Total_Visits",
    y="Rating",
    data=df
)
plt.title("Rating vs Attraction Popularity")
plt.show()


# ==============================
# 7️⃣ Rating by Visit Mode
# ==============================
if "VisitMode" in df.columns:
    plt.figure()
    sns.boxplot(x="VisitMode", y="Rating", data=df)
    plt.title("Rating Distribution by Visit Mode")
    plt.show()

print("All Visualizations Completed")