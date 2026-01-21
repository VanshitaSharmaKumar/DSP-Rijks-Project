import streamlit as st
from PIL import Image
import random
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")

st.markdown("""
<style>
/* Make Streamlit dialog wider */
div[role="dialog"] {
    width: 1100px !important;
    max-width: 95vw !important;
}

/* Make the dialog content area use available width */
div[role="dialog"] > div {
    width: 100% !important;
    max-width: 95vw !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Helpers
# ----------------------------
def get_row_field(row, col, default="—"):
    val = row.get(col, None)
    if val is None:
        return default
    s = str(val).strip()
    return s if s and s.lower() != "nan" else default


def open_details_dialog(row, reason_text=None, key="details"):
    title_text = get_row_field(row, "title", "Artwork")

    @st.dialog(title_text)
    def _dialog():
        left, right = st.columns([1.2, 1])

        # ✅ LEFT: Image
        with left:
            img_url = row.get("image_url", "")
            img_file = row.get("image_file", "")

            if isinstance(img_url, str) and img_url.strip():
                st.image(img_url, use_container_width=True)
            elif isinstance(img_file, str) and img_file and os.path.exists(img_file):
                st.image(Image.open(img_file), use_container_width=True)
            else:
                st.info("No image available.")

        # ✅ RIGHT: Details panel
        with right:
            st.markdown("### Details")

            st.write(f"**Title:** {get_row_field(row, 'title')}")
            st.write(f"**Artist:** {get_row_field(row, 'artist')}")
            st.write(f"**Year:** {get_row_field(row, 'dating')}")

            st.write(f"**Object type:** {get_row_field(row, 'object_type')}")
            st.write(f"**Museum department:** {get_row_field(row, 'department')}")

            st.write(f"**Dimensions:** {get_row_field(row, 'dimensions')}")

            st.markdown("**Description:**")
            st.write(get_row_field(row, "description"))

            st.divider()

            # ✅ AI Transparency
            st.markdown("### AI Transparency")
            st.write(
                reason_text
                or "This artwork was recommended because it is similar to your selected artwork(s) based on cosine similarity between feature vectors."
            )

    _dialog()


def build_search_text(df: pd.DataFrame) -> pd.Series:
    # Combine useful columns into one searchable string per row
    parts = []
    for col in [
        "title",
        "alt_titles",
        "description",
        "artist",
        "dating",
        "object_type",
        "materials",
        "subjects",
        "department",
    ]:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))

    if not parts:
        return pd.Series([""] * len(df))

    return (
        (" ".join([""] * 0) + parts[0]).str.lower()
        if len(parts) == 1
        else (
            parts[0]
            .astype(str)
            .str.cat([p.astype(str) for p in parts[1:]], sep=" ", na_rep="")
            .str.lower()
        )
    )


def exhibition_boost(candidate_uri, selected_uris, artwork_to_exh, weight=0.25):
    if candidate_uri not in artwork_to_exh:
        return 0.0

    shared = 0
    for uri in selected_uris:
        shared += len(artwork_to_exh.get(uri, set()) & artwork_to_exh[candidate_uri])

    # log scaling prevents domination
    return weight * np.log1p(shared)


def Reccomend_art(
    merged_final_features,
    user_selected_indices,
    df,
    artwork_to_exh,
    lambda_exh=0.25,
    k=6,
):
    selected = merged_final_features[user_selected_indices]
    sims = cosine_similarity(selected, merged_final_features)

    # Average similarity across all selected artworks
    mean_sims = sims.mean(axis=0)

    # prevent recommending selected artworks
    mean_sims[user_selected_indices] = -1

    selected_uris = df.iloc[user_selected_indices]["id"].tolist()

    boosts = np.zeros(len(df))
    for i in range(len(df)):
        boosts[i] = exhibition_boost(
            df.iloc[i]["id"], selected_uris, artwork_to_exh, weight=lambda_exh
        )

    final_scores = mean_sims + boosts

    # Sort and remove duplicates + already selected
    rec_idx = np.argsort(final_scores)[::-1]
    rec_idx = [i for i in rec_idx if i not in user_selected_indices]  # remove selected
    rec_idx = list(dict.fromkeys(rec_idx))  # remove duplicates while preserving order

    return np.array(rec_idx[:k])


# ----------------------------
# Data loading
# ----------------------------
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
    return np.load("./DATA/final_multimodal_features.npy")


@st.cache_data
def load_exhibition_df():
    return pd.read_csv("./DATA/objects_in_exhibtions.csv")


@st.cache_data
def build_exhibition_maps(exh_df):
    artwork_to_exh_ids = (
        exh_df.groupby("CollectionObject.uri")["Exhibition.nodeId"].apply(set).to_dict()
    )

    artwork_to_exh_names = (
        exh_df.groupby("CollectionObject.uri")["Exhibition.enValue"]
        .apply(lambda x: sorted(set(x.dropna())))
        .to_dict()
    )

    return artwork_to_exh_ids, artwork_to_exh_names


# ----------------------------
# UI display
# ----------------------------
def display_artworks(df, indices, header, reasons=None):
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
            if isinstance(img_url, str) and img_url.strip():
                st.image(img_url, caption=caption, use_container_width=True)
            elif isinstance(img_file, str) and os.path.exists(img_file):
                st.image(Image.open(img_file), caption=caption, use_container_width=True)
            else:
                st.write("🖼️ No image available")
                st.caption(caption)

            reason_text = reasons.get(int(idx)) if reasons else None

            # Popup button (unique key per artwork)
            if st.button("View details", key=f"details_btn_{header}_{int(idx)}"):
                open_details_dialog(
                    row, reason_text=reason_text, key=f"{header}_{int(idx)}"
                )


# ----------------------------
# Main app
# ----------------------------
st.title("Rijksmuseum Artwork Recommendation")
st.caption(
    "Select a few artworks you like, and I’ll recommend similar works from the Rijksmuseum dataset."
)

df = load_metadata_df()
df["search_text"] = build_search_text(df)

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
    rec_indices = Reccomend_art(
        merged_final_features, user_selected_indices, df, artwork_to_exh_ids, k=6
    )

    if len(rec_indices) == 0:
        st.warning("Not enough data to recommend. Try selecting different artworks.")
    else:
        display_artworks(df, rec_indices, "Recommended artworks", artwork_to_exh_names)
else:
    st.info("Select at least 1 artwork to get recommendations.")