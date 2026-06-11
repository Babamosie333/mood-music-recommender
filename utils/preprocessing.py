"""
utils/preprocessing.py
Data loading and preprocessing utilities for the mood music recommender.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def load_data(filepath="data/songs.csv"):
    """Load the songs dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} songs across {df['mood'].nunique()} moods and {df['genre'].nunique()} genres.")
    return df


def explore_data(df):
    """Print basic stats about the dataset."""
    print("\n===== Dataset Overview =====")
    print(f"Shape: {df.shape}")
    print(f"\nMood distribution:\n{df['mood'].value_counts()}")
    print(f"\nGenre distribution:\n{df['genre'].value_counts()}")
    print(f"\nFeature stats:\n{df[['energy','valence','tempo','danceability','acousticness','popularity']].describe().round(2)}")


def preprocess(df):
    """
    Preprocess the dataset:
    - Select audio feature columns
    - Scale features to [0, 1]
    - Encode mood labels to integers
    Returns: X (features), y (encoded mood labels), scaler, label_encoder
    """
    feature_cols = ['energy', 'valence', 'tempo', 'danceability', 'acousticness', 'popularity']

    X = df[feature_cols].copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    le = LabelEncoder()
    y = le.fit_transform(df['mood'])

    return X_scaled, y, scaler, le


def get_feature_cols():
    return ['energy', 'valence', 'tempo', 'danceability', 'acousticness', 'popularity']
