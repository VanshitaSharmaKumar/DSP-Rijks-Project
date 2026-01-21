import streamlit as st
from PIL import Image
import random
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

def Reccomend_art(merged_final_features, user_selected_indices, k=6):
    user_selected_features = merged_final_features[user_selected_indices]
    similarity_scores = cosine_similarity(user_selected_features, merged_final_features)

    similar_images_indices = similarity_scores.argsort(axis=1)[:, ::-1]
    recommended_images_indices = similar_images_indices[:, 1:50]  # take a bigger pool

    # flatten + dedupe + remove selected
    selected_set = set(user_selected_indices)
    flat = []
    for idx in recommended_images_indices.flatten():
        idx = int(idx)
        if idx not in selected_set and idx not in flat:
            flat.append(idx)

    # if not enough, just return what we have
    if len(flat) <= k:
        return np.array(flat, dtype=int)

    # sample k without replacement
    return np.array(random.sample(flat, k), dtype=int)

@st.cache_data
def load_metadata_df():
    df = pd.read_csv("./DATA/data_df.csv")

    # safety: ensure these exist (your build script creates them)
    for col in ["title", "artist", "dating", "image_url", "image_file"]:
        if col not in df.columns:
            df[col] = ""

    return df

@st.cache_data
def load_features_array():
    return np.load("./DATA/final_features.npy")

def display_artworks(df, indices, header):
    st.subheader(header)

    if len(indices) == 0:
        st.info("No artworks to display.")
        return

    cols = st.columns(3)

    for i, idx in enumerate(indices):
        row = df.iloc[int(idx)]

        title = str(row.get("title", ""))
        artist = str(row.get("artist", ""))
        dating = str(row.get("dating", ""))

        caption = f"{title}\n{artist} — {dating}".strip()

        img_url = str(row.get("image_url", "")).strip()
        img_file = str(row.get("image_file", "")).strip()

        with cols[i % 3]:
            if img_url:
                st.image(img_url, caption=caption, use_column_width=True)
            elif img_file and os.path.exists(img_file):
                st.image(Image.open(img_file), caption=caption, use_column_width=True)
            else:
                st.write("No image available")
                st.caption(caption)


st.title("Rijksmuseum Artwork Recommendation")
st.caption("Select a few artworks you like, and I’ll recommend similar works from the Rijksmuseum dataset.")

df = load_metadata_df()
merged_final_features = load_features_array()

# Build dropdown labels
labels = (
    df["title"].fillna("").astype(str)
    + " — "
    + df["artist"].fillna("").astype(str)
    + " ("
    + df["dating"].fillna("").astype(str)
    + ")"
).tolist()

selected_labels = st.multiselect(
    "Pick 1–3 artworks you like",
    options=labels,
    default=labels[:1] if len(labels) else [],
)

if selected_labels:
    user_selected_indices = [labels.index(x) for x in selected_labels]

    # Show selected
    display_artworks(df, user_selected_indices, "Your selected artworks")

    # Recommend
    rec_indices = Reccomend_art(merged_final_features, user_selected_indices, k=6)

    if len(rec_indices) == 0:
        st.warning("Not enough data to recommend. Try selecting different artworks.")
    else:
        display_artworks(df, rec_indices, "Recommended artworks")
else:
    st.info("Select at least 1 artwork to get recommendations.")