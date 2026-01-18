import pandas as pd
import numpy as np
import os

RIJKS_CSV = os.path.join("Data Processing", "rijksmuseum_metadata.csv")
OUT_DIR = "DATA"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(RIJKS_CSV)

# Keep only what you need + make consistent column names
data_df = pd.DataFrame({
    "id": df["identifier"],
    "title": df["title_nl"].fillna(""),
    "alt_titles": df["alternative_titles_nl"].fillna(""),
    "description": df["description_nl"].fillna(""),
    "artist": df["artist"].fillna(""),
    "dating": df["dating"].fillna(""),
    "object_type": df["object_type"].fillna(""),
    "materials": df["materials_techniques"].fillna(""),
    "subjects": df["subjects"].fillna(""),
    "department": df["museum_department"].fillna(""),
    "image_url": df["image_url"].fillna(""),
    "image_file": df["image_file"].fillna(""),
})

data_df.to_csv(os.path.join(OUT_DIR, "data_df.csv"), index=False)
print("Saved:", os.path.join(OUT_DIR, "data_df.csv"), "rows=", len(data_df))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# Build one searchable text per artwork
corpus = (
    "Titel: " + data_df["title"] + "\n" +
    "Alt: " + data_df["alt_titles"] + "\n" +
    "Beschrijving: " + data_df["description"] + "\n" +
    "Kunstenaar: " + data_df["artist"] + "\n" +
    "Datering: " + data_df["dating"] + "\n" +
    "Type: " + data_df["object_type"] + "\n" +
    "Materialen: " + data_df["materials"] + "\n" +
    "Onderwerpen: " + data_df["subjects"] + "\n" +
    "Afdeling: " + data_df["department"]
).tolist()

# TF-IDF text features (good baseline, no external downloads)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text = tfidf.fit_transform(corpus)

# Optional: add structured categorical signals
ohe = OneHotEncoder(handle_unknown="ignore")
X_cat = ohe.fit_transform(data_df[["artist", "object_type", "department"]])

# Combine and convert to dense (fine for your current dataset size)
X = np.hstack([X_text.toarray(), X_cat.toarray()]).astype(np.float32)

np.save(os.path.join(OUT_DIR, "final_features.npy"), X)
print("Saved:", os.path.join(OUT_DIR, "final_features.npy"), "shape=", X.shape)