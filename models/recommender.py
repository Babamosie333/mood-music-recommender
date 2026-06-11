"""
models/recommender.py
Content-based recommender using cosine similarity on audio features.
Also supports mood-based filtering and genre filtering.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MoodMusicRecommender:
    """
    A content-based music recommender that:
    1. Predicts mood from user-provided audio feature values, OR
    2. Accepts a direct mood label from the user
    Then ranks songs by cosine similarity to an ideal feature vector for that mood.
    """

    # Ideal audio feature profiles per mood (energy, valence, tempo_norm, danceability, acousticness, popularity_norm)
    MOOD_PROFILES = {
        "happy":     [0.80, 0.90, 0.65, 0.85, 0.10, 0.85],
        "sad":       [0.20, 0.20, 0.35, 0.30, 0.80, 0.70],
        "energetic": [0.95, 0.55, 0.80, 0.80, 0.03, 0.85],
        "calm":      [0.12, 0.55, 0.25, 0.20, 0.90, 0.75],
        "romantic":  [0.38, 0.75, 0.45, 0.50, 0.60, 0.80],
        "focused":   [0.15, 0.55, 0.30, 0.22, 0.88, 0.75],
        "angry":     [0.95, 0.28, 0.75, 0.68, 0.03, 0.85],
        "nostalgic": [0.65, 0.75, 0.60, 0.68, 0.28, 0.88],
    }

    def __init__(self, df, scaler):
        self.df     = df.copy()
        self.scaler = scaler
        self._feature_cols = ['energy', 'valence', 'tempo', 'danceability', 'acousticness', 'popularity']

        # Pre-compute scaled feature matrix for all songs
        self._X_scaled = scaler.transform(df[self._feature_cols])

    def recommend_by_mood(self, mood: str, genre: str = None, top_n: int = 5):
        """
        Recommend songs for a given mood (and optional genre) using cosine similarity
        against the mood's ideal feature profile.
        """
        mood = mood.lower()
        if mood not in self.MOOD_PROFILES:
            available = ", ".join(self.MOOD_PROFILES.keys())
            raise ValueError(f"Unknown mood '{mood}'. Choose from: {available}")

        profile = np.array(self.MOOD_PROFILES[mood]).reshape(1, -1)

        # Filter by genre if provided
        mask = self.df['mood'] == mood
        if genre:
            mask &= self.df['genre'].str.lower() == genre.lower()

        subset_df  = self.df[mask].copy()
        subset_X   = self._X_scaled[mask]

        if subset_df.empty:
            print(f"  No songs found for mood='{mood}'" + (f", genre='{genre}'" if genre else "") + ". Showing all moods.")
            subset_df = self.df.copy()
            subset_X  = self._X_scaled

        sims = cosine_similarity(profile, subset_X)[0]
        subset_df = subset_df.copy()
        subset_df['similarity'] = sims
        top = subset_df.nlargest(top_n, 'similarity')[
            ['title', 'artist', 'genre', 'mood', 'energy', 'valence', 'similarity']
        ].reset_index(drop=True)
        top.index += 1
        return top

    def recommend_by_features(self, energy: float, valence: float, tempo: float,
                               danceability: float, acousticness: float,
                               popularity: float = 80, top_n: int = 5):
        """
        Given raw feature values, find the most similar songs across ALL moods.
        Also predicts the closest mood.
        """
        raw = np.array([[energy, valence, tempo, danceability, acousticness, popularity]])
        scaled = self.scaler.transform(raw)

        sims = cosine_similarity(scaled, self._X_scaled)[0]
        df = self.df.copy()
        df['similarity'] = sims
        top = df.nlargest(top_n, 'similarity')[
            ['title', 'artist', 'genre', 'mood', 'energy', 'valence', 'similarity']
        ].reset_index(drop=True)
        top.index += 1

        # Predict mood
        best_mood, best_sim = None, -1
        for mood, profile in self.MOOD_PROFILES.items():
            s = cosine_similarity(scaled, np.array(profile).reshape(1, -1))[0][0]
            if s > best_sim:
                best_sim, best_mood = s, mood

        return top, best_mood

    def get_available_genres(self, mood: str = None):
        if mood:
            return sorted(self.df[self.df['mood'] == mood]['genre'].unique())
        return sorted(self.df['genre'].unique())
