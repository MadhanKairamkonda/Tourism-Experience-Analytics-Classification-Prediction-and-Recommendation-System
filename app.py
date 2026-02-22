import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Hybrid Tourism Recommender", layout="wide")

st.title("🌍 Hybrid Tourism Recommendation System")
st.write("Collaborative Filtering + Content-Based Model")

# ==========================================================
# SAFE MODEL LOADER
# ==========================================================
@st.cache_resource
def load_models():

    model_path = "model.pkl"
    data_path = "data.pkl"

    if not os.path.exists(model_path):
        st.error("model.pkl not found. Run modelbuilding.py first.")
        st.stop()

    if not os.path.exists(data_path):
        st.error("data.pkl not found. Run modelbuilding.py first.")
        st.stop()

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    with open(data_path, "rb") as f:
        train_df = pickle.load(f)

    return model_data, train_df


model_data, train_df = load_models()

# ==========================================================
# UNPACK MODEL
# ==========================================================
best_reg = model_data["best_reg"]
P = model_data["P"]
Q = model_data["Q"]
user_map = model_data["user_map"]
item_map = model_data["item_map"]
X_columns = model_data["X_columns"]

# ==========================================================
# MF PREDICTION
# ==========================================================
def mf_predict(user_id, item_id):
    if user_id not in user_map or item_id not in item_map:
        return train_df["Rating"].mean()
    
    u = user_map[user_id]
    i = item_map[item_id]
    
    return np.dot(P[u], Q[i])


# ==========================================================
# CONTENT PREDICTION
# ==========================================================
def content_predict(base_row):

    row_df = pd.DataFrame([base_row])
    row_encoded = pd.get_dummies(row_df)

    row_encoded = row_encoded.reindex(columns=X_columns, fill_value=0)

    return best_reg.predict(row_encoded)[0]


# ==========================================================
# HYBRID SCORE
# ==========================================================
def hybrid_score(user_id, item_id, base_row, alpha=0.7):

    mf_score = mf_predict(user_id, item_id)
    content_score = content_predict(base_row)

    # Cold start case
    if user_id not in user_map or item_id not in item_map:
        return content_score

    return alpha * mf_score + (1 - alpha) * content_score


# ==========================================================
# UI CONTROLS
# ==========================================================
st.sidebar.header("Controls")

user_list = sorted(train_df["UserId"].unique())
selected_user = st.sidebar.selectbox("Select User", user_list)

top_n = st.sidebar.slider("Number of Recommendations", 3, 15, 5)
alpha = st.sidebar.slider("Hybrid Weight (CF vs Content)", 0.0, 1.0, 0.7)

# ==========================================================
# GENERATE RECOMMENDATIONS
# ==========================================================
if st.button("Generate Recommendations"):

    rated_items = train_df[train_df["UserId"] == selected_user]["AttractionId"].values
    all_items = train_df["AttractionId"].unique()

    recommendations = []

    for item in all_items:

        if item in rated_items:
            continue

        item_rows = train_df[train_df["AttractionId"] == item]

        if item_rows.empty:
            continue

        base_row = item_rows.iloc[0].to_dict()

        score = hybrid_score(selected_user, item, base_row, alpha)

        if np.isnan(score):
            continue

        recommendations.append((item, score))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recs = recommendations[:top_n]

    st.subheader("Top Recommendations")

    if not top_recs:
        st.warning("No recommendations available.")
    else:
        for item, score in top_recs:

            row = train_df[train_df["AttractionId"] == item]

            if not row.empty and "AttractionName" in train_df.columns:
                attraction_name = row["AttractionName"].iloc[0]
            else:
                attraction_name = f"Attraction {item}"

            st.write(f"**{attraction_name}**  — Predicted Rating: {score:.2f}")