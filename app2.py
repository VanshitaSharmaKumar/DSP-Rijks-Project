import streamlit as st
from PIL import Image
import random
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# from scipy.sparse import load_npz

def build_search_text(df: pd.DataFrame) -> pd.Series:
    # Combine useful columns into one searchable string per row
    parts = []
    for col in ["title", "alt_titles", "description", "artist", "dating", "object_type", "materials", "subjects", "department"]:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    if not parts:
        return pd.Series([""] * len(df))

    return (" ".join([""] * 0) + parts[0]).str.lower() if len(parts) == 1 else (
        parts[0].astype(str)
        .str.cat([p.astype(str) for p in parts[1:]], sep=" ", na_rep="")
        .str.lower()
    )

# def Reccomend_art(merged_final_features, user_selected_indices, k=6):
#     user_selected_features = merged_final_features[user_selected_indices]
#     similarity_scores = cosine_similarity(user_selected_features, merged_final_features)

#     similar_images_indices = similarity_scores.argsort(axis=1)[:, ::-1]
#     recommended_images_indices = similar_images_indices[:, 1:50]  # take a bigger pool

#     # flatten + dedupe + remove selected
#     selected_set = set(user_selected_indices)
#     flat = []
#     for idx in recommended_images_indices.flatten():
#         idx = int(idx)
#         if idx not in selected_set and idx not in flat:
#             flat.append(idx)

#     # if not enough, just return what we have
#     if len(flat) <= k:
#         return np.array(flat, dtype=int)

#     # sample k without replacement
#     return np.array(random.sample(flat, k), dtype=int)

# def Reccomend_art(merged_final_features, user_selected_indices, k=6):
#     selected = merged_final_features[user_selected_indices]  # (m, d)
#     sims = cosine_similarity(selected, merged_final_features)  # (m, n)

#     # Average similarity across all selected artworks
#     mean_sims = sims.mean(axis=0)  # (n,)

#     # Do not recommend already selected
#     mean_sims[user_selected_indices] = -1

#     # Top-k most similar, deterministic
#     rec_idx = np.argsort(mean_sims)[::-1][:k]
#     return rec_idx.astype(int)

def Reccomend_art(merged_final_features, user_selected_indices, df, artwork_to_exh, lambda_exh=0.25, k=6):
    selected = merged_final_features[user_selected_indices]
    sims = cosine_similarity(selected, merged_final_features)

    # Average similarity across all selected artworks
    mean_sims = sims.mean(axis=0)
    
    # prevent recommending selected artworks
    mean_sims[user_selected_indices] = -1

    selected_uris = df.iloc[user_selected_indices]["id"].tolist()
    
    boosts = np.zeros(len(df))
    for i in range(len(df)):
        boosts[i] = exhibition_boost(df.iloc[i]["id"], selected_uris, artwork_to_exh, weight=lambda_exh)
    
    final_scores = mean_sims + boosts
    
    # Sort and remove duplicates + already selected
    rec_idx = np.argsort(final_scores)[::-1]
    rec_idx = [i for i in rec_idx if i not in user_selected_indices]  # remove selected
    rec_idx = list(dict.fromkeys(rec_idx))  # remove duplicates while preserving order
    
    return np.array(rec_idx[:k])

def exhibition_boost(candidate_uri, selected_uris, artwork_to_exh, weight=0.25):
    if candidate_uri not in artwork_to_exh:
        return 0.0

    shared = 0
    for uri in selected_uris:
        shared += len(
            artwork_to_exh.get(uri, set())
            & artwork_to_exh[candidate_uri]
        )

    # log scaling prevents domination
    return weight * np.log1p(shared)

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

@st.cache_data
def load_exhibition_df():
    return pd.read_csv("./DATA/objects_in_exhibtions.csv")

# Exhibition mappings
@st.cache_data
def build_exhibition_maps(exh_df):
    artwork_to_exh_ids = (
        exh_df
        .groupby("CollectionObject.uri")["Exhibition.nodeId"]
        .apply(set)
        .to_dict()
    )

    artwork_to_exh_names = (
        exh_df
        .groupby("CollectionObject.uri")["Exhibition.enValue"]
        .apply(lambda x: sorted(set(x.dropna())))
        .to_dict()
    )

    return artwork_to_exh_ids, artwork_to_exh_names

def display_artworks(df, indices, header, artwork_to_exh_names=None):
    st.subheader(header)
    cols = st.columns(3)

    for i, idx in enumerate(indices):
        row = df.iloc[int(idx)]

        title = str(row.get("title", "")).strip()
        artist = str(row.get("artist", "")).strip()
        dating = str(row.get("dating", "")).strip()
        caption = f"{title}\n{artist} — {dating}".strip()

        img_url = row.get("image_url")
        img_file = row.get("image_file")

        with cols[i % 3]:
            # Display image
            if isinstance(img_url, str) and img_url.strip():
                st.image(img_url, caption=caption, width="stretch")
            elif isinstance(img_file, str) and os.path.exists(img_file):
                st.image(Image.open(img_file), caption=caption, width="stretch")
            else:
                st.write("🖼️ No image available")
                st.caption(caption)

            # # Display exhibition info
            # if artwork_to_exh_names:
            #     exhibitions = artwork_to_exh_names.get(row["id"], [])
            #     if exhibitions:
            #         st.caption("🖼️ Exhibited in: " + "; ".join(exhibitions[:3]))

st.title("Rijksmuseum Artwork Recommendation")
st.caption("Select a few artworks you like, and I’ll recommend similar works from the Rijksmuseum dataset.")

df = load_metadata_df()
df["search_text"] = build_search_text(df)   # add this line
merged_final_features = load_features_array()

# Add exhibition data to app
exh_df = load_exhibition_df()
artwork_to_exh_ids, artwork_to_exh_names = build_exhibition_maps(exh_df)

# Build dropdown labels
labels = (
    df["title"].fillna("").astype(str)
    + " — "
    + df["artist"].fillna("").astype(str)
    + " ("
    + df["dating"].fillna("").astype(str)
    + ")"
).tolist()

# Keyword search input
query = st.text_input(
    "Search artworks (title/description/subjects/etc.)",
    placeholder="Try: flower, portrait, landscape, japan, vase...",
).strip().lower()

# Filter dropdown options based on query
if query:
    mask = df["search_text"].str.contains(query, regex=False, na=False)
    filtered_indices = df.index[mask].tolist()
    filtered_labels = [labels[i] for i in filtered_indices]
    st.caption(f"Matches: {len(filtered_labels)}")
else:
    filtered_labels = labels

selected_labels = st.multiselect(
    "Pick 1–3 artworks you like",
    options=filtered_labels,
    default=[],
)

if selected_labels:
    user_selected_indices = [labels.index(x) for x in selected_labels]

    # Show selected
    display_artworks(df, user_selected_indices, "Your selected artworks", artwork_to_exh_names)

    # Recommend
    rec_indices = Reccomend_art(merged_final_features, user_selected_indices, df, artwork_to_exh_ids, k=6)

    if len(rec_indices) == 0:
        st.warning("Not enough data to recommend. Try selecting different artworks.")
    else:
        display_artworks(df, rec_indices, "Recommended artworks", artwork_to_exh_names)
else:
    st.info("Select at least 1 artwork to get recommendations.")