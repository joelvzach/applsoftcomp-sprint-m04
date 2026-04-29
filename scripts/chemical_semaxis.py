import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import os

# import data set
df = pd.read_csv('data/chemicals.csv')

# create sentence transformer model
model = SentenceTransformer("all-mpnet-base-v2")  

def make_axis(positive_words, negative_words, embedding_model):
    pos_emb = embedding_model.encode(positive_words, normalize_embeddings=True)
    neg_emb = embedding_model.encode(negative_words, normalize_embeddings=True)

    pole_pos = pos_emb.mean(axis=0)
    pole_neg = neg_emb.mean(axis=0)

    v = pole_pos - pole_neg
    v = v / (np.linalg.norm(v) + 1e-10)

    return v / (np.linalg.norm(v) + 1e-10)

def score_words(words, axis, embedding_model):
    emb = embedding_model.encode(list(words), normalize_embeddings=True)
    proj = emb @ axis
    return proj

# create first axis (safety)
axis1_pos = [
    'safe in small doses',
    'medical',
    'harmless',
    'benign',
    'low-risk',
]

axis1_neg = [
    'toxic',
    'reactive',
    'lethal',
    'hazardous',
    'dangerous',
    'harmful',
]
axis_safety = make_axis(axis1_pos, axis1_neg, model)

# create second axis (utility)
axis2_pos = [
    'Nutrient',
    'Biological',
    'Metabolic',
    'Organic',
    'Vital',
    'Life'
]
axis2_neg = [
    'Industrial',
    'Synthetic',
    'Inorganic',
    'Artificial',
    'Mechanical',
    'Mineral'
]
axis_utility = make_axis(axis2_pos, axis2_neg, model)

# score the chemicals
x = score_words(df["name"].tolist(), axis_safety, model)
y = score_words(df["name"].tolist(), axis_utility, model)
df_scored = df.assign(x=x, y=y)

# --- NEW: color by class ---
classes = df_scored["class"].unique()
cmap = plt.get_cmap("tab10")  # good categorical colormap

class_to_color = {cls: cmap(i % 10) for i, cls in enumerate(classes)}

# Create figure
plt.figure(figsize=(12, 10))

# Plot each class separately (so legend works cleanly)
for cls in classes:
    subset = df_scored[df_scored["class"] == cls]
    plt.scatter(
        subset["x"],
        subset["y"],
        label=cls,
        color=class_to_color[cls],
        alpha=0.7
    )

# Label each point (optional: still cluttered for large datasets)
for i, txt in enumerate(df_scored["name"]):
    plt.annotate('   ' + txt,
                 (df_scored["x"][i], df_scored["y"][i]),
                 fontsize=5,
                 alpha=0.6)

# Axis labels and title
plt.xlabel("Safety Axis (Hazardous  →  Safe)")
plt.ylabel("Utility Axis (Industrial  →  Biological)")
plt.title("Chemical SemAxis Visualization")

# Reference lines
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

# Legend
plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')

# Grid + layout
plt.grid(True)
plt.tight_layout()

# Save figure
output_dir = "../figs"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "chemical_semaxis.png"),
            dpi=300,
            bbox_inches="tight")

plt.show()