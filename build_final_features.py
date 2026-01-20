import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import normalize

# Load metadata and text features
df = pd.read_csv("DATA/data_df.csv")
X_text = np.load("DATA/final_features.npy")
X_text = normalize(X_text) 

# Load image features
img_data = torch.load("image_features_2.pt", weights_only=False)
X_img = img_data["features"].numpy()
img_ids = img_data["ids"]
X_img = normalize(X_img) # normalize so one modality doesnt dominate the other

# id to image index map
img_id_to_idx = {id_: i for i, id_ in enumerate(img_ids)}

# Align features
text_feats = []
img_feats = []

for i, row in df.iterrows():
    text_feats.append(X_text[i])
    if row["id"] in img_id_to_idx:
        img_feats.append(X_img[img_id_to_idx[row["id"]]])
    else:
        img_feats.append(np.zeros(2048))  # no image fallback

X_final = np.hstack([text_feats, img_feats]).astype(np.float32)

np.save("DATA/final_multimodal_features.npy", X_final)