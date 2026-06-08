import os
import pickle
import pandas as pd
import streamlit as st

# Setup robust relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, 'data', 'synthetic_recommendation_data.csv')
knn_path = os.path.join(BASE_DIR, 'src', 'data', 'models', 'knn_model.pkl')
lr_path = os.path.join(BASE_DIR, 'src', 'data', 'models', 'lr_model.pkl')

# Load data + models
dataset = pd.read_csv(dataset_path)
knn = pickle.load(open(knn_path, "rb"))
lr = pickle.load(open(lr_path, "rb"))

# UI
st.title("Smart Recommendation System")

user_id = st.selectbox("Select User", dataset["user_id"].unique())

# Prepare user features matching training features: price, time_spent, clicked
user_features = dataset.groupby("user_id")[["price", "time_spent", "clicked"]].mean()

# Get similar users (use double brackets to keep DataFrame structure and feature names)
distances, indices = knn.kneighbors(user_features.loc[[user_id]])

similar_users = indices[0]

# Candidate items
candidate_items = dataset[dataset["user_id"].isin(similar_users)].copy()

# Predict click probability using the trained pipeline
X = candidate_items[["price", "time_spent", "category"]]
candidate_items["score"] = lr.predict_proba(X)[:, 1]

# Top 5
recommendations = candidate_items.sort_values("score", ascending=False).head(5)

st.write("Top Recommendations:")
st.dataframe(recommendations[["item_id", "category", "price", "score"]])