import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sentence_transformers import SentenceTransformer
from drawdata import ScatterWidget

# import data set
data = pd.read_csv('../data/chemicals.csv')

# create sentence transformer model
model = SentenceTransformer("all-mpnet-base-v2")  

def make_axis(positive_words, negative_words, embedding_model):
    """Return a unit-length semantic axis from two word sets."""

    # get the embeddings for each pole
    pos_emb = embedding_model.encode(positive_words, normalize_embeddings=True)
    neg_emb = embedding_model.encode(negative_words, normalize_embeddings=True)

    # Compute the pole centroids
    # axis = 0 means "average across the rows, keep the columns (dims) intact"
    # since pos_emb is shape (num_pos_words, embedding_dim), the mean is shape (embedding_dim,)
    pole_pos = pos_emb.mean(axis=0)  # (embedding_dim,)
    pole_neg = neg_emb.mean(axis=0)  # (embedding_dim,)

    # The axis is the difference between the two centroids, normalized to unit length.
    v = pole_pos - pole_neg

    v = v / (np.linalg.norm(v) + 1e-10)  # add small epsilon to prevent division by zero

    return v / (np.linalg.norm(v) + 1e-10)

def score_words(words, axis, embedding_model):
    """Project each word onto the axis. Returns one score per word."""

    emb = embedding_model.encode(list(words), normalize_embeddings=True)

    # Projection to the axis is just a dot product (since the axis is unit-length).
    # @ is matrix multiplication in NumPy. Since `emb` is shape (num_words, embedding_dim) and `axis` is shape (embedding_dim,), the result is shape (num_words,), which is exactly what we want: one score per word.
    proj = emb @ axis

    return proj

# create first axis for stability of chemicals i.e. dangerous or safe
axis1_pos = [
    'Explosive',
    'Toxic',
    'Corrosive',
    'Lethal',
    'Hazardous',
    'Reactive'
]

axis1_neg = [
    'Inert',
    'Stable',
    'Harmless',
    'Edible',
    'Safe',
    'Benign'
]
axis_stability = make_axis(axis1_pos, axis1_neg, model)

# create second axis for utility of chemicals i.e. biological or industrial use
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

