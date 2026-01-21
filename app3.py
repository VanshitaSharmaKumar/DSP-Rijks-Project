import streamlit as st
from PIL import Image
import random
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
# import joblib
# from scipy.sparse import load_npz
from openai import OpenAI
import textwrap


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

def recommend_with_details(merged_final_features, user_selected_indices, df, artwork_to_exh_ids, artwork_to_exh_names, lambda_exh=0.25, k=6):
    selected = merged_final_features[user_selected_indices]
    sims = cosine_similarity(selected, merged_final_features)  # (m, n)
    mean_sims = sims.mean(axis=0)  # (n,)

    # prevent recommending selected artworks
    mean_sims[user_selected_indices] = -1

    selected_uris = df.iloc[user_selected_indices]["id"].tolist()

    boosts = np.zeros(len(df))
    shared_counts = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        cand_uri = df.iloc[i]["id"]
        if cand_uri in artwork_to_exh_ids:
            shared = 0
            cand_set = artwork_to_exh_ids[cand_uri]
            for uri in selected_uris:
                shared += len(artwork_to_exh_ids.get(uri, set()) & cand_set)
            shared_counts[i] = shared
            boosts[i] = lambda_exh * np.log1p(shared)
        else:
            shared_counts[i] = 0
            boosts[i] = 0.0

    final_scores = mean_sims + boosts

    rec_idx = np.argsort(final_scores)[::-1]
    rec_idx = [i for i in rec_idx if i not in user_selected_indices]
    rec_idx = list(dict.fromkeys(rec_idx))
    rec_idx = rec_idx[:k]

    details = {}
    for i in rec_idx:
        uri = df.iloc[i]["id"]
        # take some exhibition names if available
        exh_names = artwork_to_exh_names.get(uri, []) if artwork_to_exh_names else []
        details[int(i)] = {
            "mean_similarity": float(mean_sims[i]),
            "exhibition_boost": float(boosts[i]),
            "shared_exhibitions_count": int(shared_counts[i]),
            "exhibition_names": exh_names[:5],
            "final_score": float(final_scores[i]),
        }

    return np.array(rec_idx, dtype=int), details

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

@st.cache_data(show_spinner=False)
def generate_recommendation_explanation(
    selected_rows: list[dict],
    recommended_row: dict,
    rec_detail: dict,
    curator_goal: str,
    model_name: str,
):
    # Load key safely from Streamlit secrets/env
    api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        return "⚠️ OpenAI key missing. Add OPENAI_API_KEY to Streamlit secrets (or environment variables)."

    client = OpenAI(api_key=api_key)

    # Build a transparent, grounded prompt (no mystery claims)
    prompt = f"""
You are a museum curator assistant. Explain WHY a recommended artwork was suggested to the user.

Constraints:
- Be specific and grounded ONLY in the provided signals and metadata.
- Do NOT invent facts about the artwork that are not in the metadata.
- Explain in 2 parts:
  "Why it matches" (bullets, 3–6 bullets)
  "Notes" (brief: what signals were used + limitations)

Curator goal: {curator_goal}

User-selected artworks metadata (list):
{selected_rows}

Recommended artwork metadata:
{recommended_row}

Recommendation signals (computed):
- mean_cosine_similarity: {rec_detail.get("mean_similarity")}
- exhibition_boost: {rec_detail.get("exhibition_boost")}
- shared_exhibitions_count: {rec_detail.get("shared_exhibitions_count")}
- exhibition_names: {rec_detail.get("exhibition_names")}
- final_score: {rec_detail.get("final_score")}
""".strip()

    resp = client.responses.create(
        model=model_name,
        input=prompt,
    )
    # Responses API returns output text; simplest extraction:
    return resp.output_text

@st.dialog("Artwork details")
def artwork_popup(title, img_url, img_file, caption, explanation_text):
    left, right = st.columns([1, 1])

    with left:
        if isinstance(img_url, str) and img_url.strip():
            st.image(img_url, caption=caption, width="stretch")
        elif isinstance(img_file, str) and os.path.exists(img_file):
            st.image(Image.open(img_file), caption=caption, width="stretch")
        else:
            st.write("🖼️ No image available")
            st.caption(caption)

    with right:
        st.markdown("### Why this was recommended")
        st.write(explanation_text)

        # Optional: close button (dialog also has an X)
        if st.button("Close"):
            st.rerun()


def display_artworks(
    df,
    indices,
    header,
    artwork_to_exh_names=None,
    *,
    mode="generic",  # "selected" or "recommended" or "generic"
    rec_details=None,  # dict idx -> detail
    selected_indices=None,  # list of selected indices (needed for explanation)
    curator_goal="Help me build a coherent exhibition from the chosen artworks.",
    openai_model="gpt-4o",
):
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

            # Only show "Explain" on recommended items
            if mode == "recommended":
                btn_key = f"explain_{header}_{int(idx)}"
                if st.button("🔎 Why recommended?", key=btn_key):
                    if not selected_indices:
                        explanation = "Select artworks first to generate an explanation."
                    else:
                        # Prepare grounded inputs
                        selected_rows = []
                        for sidx in selected_indices:
                            srow = df.iloc[int(sidx)]
                            selected_rows.append({
                                "id": srow.get("id", ""),
                                "title": str(srow.get("title", "")).strip(),
                                "artist": str(srow.get("artist", "")).strip(),
                                "dating": str(srow.get("dating", "")).strip(),
                                "object_type": str(srow.get("object_type", "")).strip(),
                                "materials": str(srow.get("materials", "")).strip(),
                                "subjects": str(srow.get("subjects", "")).strip(),
                                "department": str(srow.get("department", "")).strip(),
                            })

                        recommended_row = {
                            "id": row.get("id", ""),
                            "title": title,
                            "artist": artist,
                            "dating": dating,
                            "object_type": str(row.get("object_type", "")).strip(),
                            "materials": str(row.get("materials", "")).strip(),
                            "subjects": str(row.get("subjects", "")).strip(),
                            "department": str(row.get("department", "")).strip(),
                            "description": str(row.get("description", "")).strip(),
                        }

                        detail = (rec_details or {}).get(int(idx), {})
                        explanation = generate_recommendation_explanation(
                            selected_rows=selected_rows,
                            recommended_row=recommended_row,
                            rec_detail=detail,
                            curator_goal=curator_goal,
                            model_name=openai_model,
                        )

                    artwork_popup(title, img_url, img_file, caption, explanation)

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
    # Sidebar controls (optional but useful)
    with st.sidebar:
        st.markdown("## Explanation settings")
        curator_goal = st.text_area(
            "What is the exhibition goal?",
            value="Help me build a coherent exhibition from the chosen artworks.",
            height=100,
        )
        openai_model = st.text_input("OpenAI model", value="gpt-4.1-mini")

    rec_indices, rec_details = recommend_with_details(
        merged_final_features,
        user_selected_indices,
        df,
        artwork_to_exh_ids,
        artwork_to_exh_names,
        lambda_exh=0.25,
        k=6,
    )

    if len(rec_indices) == 0:
        st.warning("Not enough data to recommend. Try selecting different artworks.")
    else:
        display_artworks(
            df,
            rec_indices,
            "Recommended artworks",
            artwork_to_exh_names,
            mode="recommended",
            rec_details=rec_details,
            selected_indices=user_selected_indices,
            curator_goal=curator_goal,
            openai_model=openai_model,
        )

else:
    st.info("Select at least 1 artwork to get recommendations.")