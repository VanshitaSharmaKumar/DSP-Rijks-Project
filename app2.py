import streamlit as st
from PIL import Image
import random
import numpy as np
import pandas as pd
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download


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

        # LEFT: Image
        with left:
            img_url = row.get("image_url", "")
            img_file = row.get("image_file", "")

            if isinstance(img_url, str) and img_url.strip():
                st.image(img_url, use_container_width=True)
            elif isinstance(img_file, str) and img_file and os.path.exists(img_file):
                st.image(Image.open(img_file), use_container_width=True)
            else:
                st.info("No image available.")

        # RIGHT: Details panel
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

            # AI Transparency
            if reason_text:
                st.markdown("### AI Transparency")
                st.write(reason_text)

            #st.markdown("### AI Transparency")
            #st.write(
            #    reason_text
            #   or "This artwork was recommended because it is similar to your selected artwork(s) based on cosine similarity between feature vectors."
            #)

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


def Recommend_art(
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


def _to_set(value) -> set:
    """Convert metadata cell to a set of tokens (robust to NaN, lists, etc.)."""
    if value is None:
        return set()
    if isinstance(value, float) and np.isnan(value):
        return set()
    if isinstance(value, (set, list, tuple)):
        return {str(x).strip().lower() for x in value if str(x).strip()}

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return set()

    # split on common separators used in CSV fields
    parts = re.split(r"[;|,]", s)
    return {p.strip().lower() for p in parts if p.strip()}


def build_reason_text(df, features, rec_idx: int, selected_indices: list[int]) -> str:
    """
    Create an explainable reason for a recommended artwork.
    Uses cosine similarity + shared metadata (if available).
    """
    selected_feats = features[selected_indices]              # (m, d)
    rec_feat = features[int(rec_idx)].reshape(1, -1)         # (1, d)

    sims = cosine_similarity(rec_feat, selected_feats).flatten()  # (m,)
    best_pos = int(np.argmax(sims))
    best_sel_idx = int(selected_indices[best_pos])
    best_score = float(sims[best_pos])

    rec_row = df.iloc[int(rec_idx)]
    sel_row = df.iloc[int(best_sel_idx)]

    shared_bits = []
    for col, label in [
        ("subjects", "subjects"),
        ("materials", "materials"),
        ("object_type", "object type"),
        ("department", "museum department"),
    ]:
        if col in df.columns:
            rec_set = _to_set(rec_row.get(col))
            sel_set = _to_set(sel_row.get(col))
            shared = sorted(rec_set.intersection(sel_set))
            if shared:
                shared_bits.append(f"shared {label}: {', '.join(shared[:3])}")

    sel_title = str(sel_row.get("title", "")).strip()
    sel_artist = str(sel_row.get("artist", "")).strip()

    if sel_title and sel_artist:
        sel_ref = f"“{sel_title}” ({sel_artist})"
    elif sel_title:
        sel_ref = f"“{sel_title}”"
    else:
        sel_ref = "one of your selected artworks"

    base = (
        f"This artwork was recommended because it is most similar to {sel_ref} "
        f"(Similarity {best_score:.2f})."
    )
    if shared_bits:
        base += " It also has " + "; ".join(shared_bits) + "."
    return base

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_metadata_df():
    df = pd.read_csv("./DATA/data_df.csv")

    # safety: ensure these exist 
    for col in ["title", "artist", "dating", "image_url", "image_file"]:
        if col not in df.columns:
            df[col] = ""

    return df


@st.cache_data
def load_features_array():
    HF_REPO_ID = "sofiaspero/rijks_features"
    file_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="final_multimodal_features.npy"
    )
    return np.load(file_path)


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
def clean_field(val, default="Unknown", blacklist=None):
    """Return default if value is missing, empty, NaN, or in blacklist."""
    if pd.isna(val) or str(val).strip() == "" or str(val).strip().lower() in (b.lower() for b in (blacklist or [])):
        return default
    return str(val).strip()

def build_caption(row):
    title = clean_field(row.get("title"), "Untitled")
    artist = clean_field(row.get("artist"), "Unknown", blacklist=["nan", "anoniem"])
    dating = clean_field(row.get("dating"), "Unknown")

    # Avoid duplicating year if it appears in title
    year_in_title = re.search(r"\b\d{4}\b", title)
    if year_in_title:
        caption = f"**{title}**\nby {artist}"
    else:
        caption = f"**{title}**\nby {artist} ({dating})"
    return caption

def display_artworks(df, indices, header, reasons=None, allow_add_to_collection=True, collection_cols=3, show_add_button=True):
    """
    Display artworks in a grid with captions, view details button,
    and optional add-to-collection button.
    
    Parameters:
    - df: DataFrame with artwork metadata
    - indices: list of indices to display
    - header: section header
    - reasons: optional dict with recommendation reasons
    - allow_add_to_collection: whether to show add button
    - collection_cols: number of columns in grid
    - show_add_button: whether to display add-to-collection button (False for 'My collection')
    """
    st.subheader(header)
    cols = st.columns(collection_cols)

    for i, idx in enumerate(indices):
        row = df.iloc[int(idx)]
        caption = build_caption(row)
        img_url = row.get("image_url")
        img_file = row.get("image_file")

        with cols[i % collection_cols]:
            # Show image
            if isinstance(img_url, str) and img_url.strip():
                st.image(img_url, caption=caption, use_container_width=True)
            elif isinstance(img_file, str) and os.path.exists(img_file):
                st.image(img_file, caption=caption, use_container_width=True)
            else:
                st.write("No image available")
                st.caption(caption)

            # Optional reason for recommendation
            reason_text = reasons.get(int(idx)) if reasons else None

            # Buttons row
            col1, col2 = st.columns([1, 1])  # tight columns for buttons

            # View details button
            with col1:
                if st.button("View details", key=f"details_btn_{header}_{idx}"):
                    open_details_dialog(row, reason_text=reason_text, key=f"{header}_{idx}")

            # Add to collection button (only if allowed and not in collection)
            if allow_add_to_collection and show_add_button:
                with col2:
                    if idx not in st.session_state.curator_collection:
                        add_key = f"add_btn_{header}_{idx}"
                        if st.button("Add to collection", key=add_key):
                            st.session_state.curator_collection.append(idx)
                            st.success(f"Added '{build_caption(row)}' to my collection")


# ----------------------------
# Main app
# ----------------------------
st.title("Rijksmuseum Artwork Recommendation")
st.caption(
    "Select a few artworks you like, and I’ll recommend similar works from the Rijksmuseum dataset."
)

# Session state for curator collection
if "curator_collection" not in st.session_state:
    st.session_state.curator_collection = []

#df = load_metadata_df()
#df["search_text"] = build_search_text(df)
#merged_final_features = load_features_array()

df = load_metadata_df()

# Identify duplicates
dup_mask = df.duplicated(subset=["title", "artist", "dating"], keep="first")

# Drop duplicates from df
df = df[~dup_mask].reset_index(drop=True)

# Drop corresponding rows from features array
merged_final_features = load_features_array()
merged_final_features = merged_final_features[~dup_mask.values]

# Add exhibition data to app
exh_df = load_exhibition_df()
artwork_to_exh_ids, artwork_to_exh_names = build_exhibition_maps(exh_df)

# Build search text for keyword filtering
df["search_text"] = build_search_text(df)

# Build dropdown labels using cleaned captions
dropdown_labels = [
    f"{clean_field(df.iloc[i].get('title'), 'Untitled')} by "
    f"{clean_field(df.iloc[i].get('artist'), 'Unknown', blacklist=['nan', 'anoniem'])} "
    f"({clean_field(df.iloc[i].get('dating'), 'Unknown')})"
    for i in range(len(df))
]
caption_to_index = {dropdown_labels[i]: i for i in range(len(df))}

# Keyword search input
query = st.text_input(
    "Search artworks (title/description/subjects/etc.)",
    placeholder="Try: flower, portrait, landscape, japan, vase...",
).strip().lower()

# Filter dropdown options based on query
if query:
    keywords = query.split()  # split by spaces into separate words
    mask = pd.Series(True, index=df.index)
    
    for word in keywords:
        mask &= df["search_text"].str.contains(word, regex=False, na=False)
    
    filtered_indices = df.index[mask].tolist()
    filtered_labels = [dropdown_labels[i] for i in filtered_indices]
    st.caption(f"Matches: {len(filtered_labels)}")
else:
    filtered_labels = dropdown_labels

selected_labels = st.multiselect(
    "Pick 1–3 artworks you like",
    options=filtered_labels,
    default=[],
)

if selected_labels:
    # Map captions back to indices
    user_selected_indices = [caption_to_index[label] for label in selected_labels]

    # Show selected artworks (no reasons needed)
    display_artworks(df, user_selected_indices, "Your selected artworks")

    rec_indices = Recommend_art(
        merged_final_features, user_selected_indices, df, artwork_to_exh_ids, k=6
    )

    reasons = {}  # always defined

    if len(rec_indices) == 0:
        st.warning("Not enough data to recommend. Try selecting different artworks.")
    else:
        reasons = {
            int(idx): build_reason_text(df, merged_final_features, int(idx), user_selected_indices)
            for idx in rec_indices
        }

    display_artworks(
        df,
        rec_indices,
        "Recommended artworks",
        reasons=reasons,
        allow_add_to_collection=True
    )
else:
    st.info("Select at least 1 artwork to get recommendations.")

# --- Always show curator collection ---
if st.session_state.curator_collection:
    display_artworks(df, st.session_state.curator_collection, "My collection", collection_cols=4, allow_add_to_collection=False, show_add_button=False)

