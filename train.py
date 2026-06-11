"""
train.py
Full training pipeline:
  1. Load & explore data
  2. Preprocess
  3. Train & compare models
  4. Save best model
  5. Generate all visualizations
"""

import os
import sys

# Make sure sibling packages resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import load_data, explore_data, preprocess
from models.classifier   import train_all_models, save_model
from models.recommender  import MoodMusicRecommender
from utils.visualize     import (
    plot_confusion_matrix, plot_feature_importance,
    plot_mood_clusters, plot_recommendation_scores,
    plot_model_comparison,
)


def main():
    print("=" * 60)
    print("  Mood-Based Music Recommender — Training Pipeline")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    df = load_data("data/songs.csv")
    explore_data(df)

    # ── 2. Preprocess ─────────────────────────────────────────
    X, y, scaler, le = preprocess(df)

    # ── 3. Train & evaluate ───────────────────────────────────
    best_model, best_name, results_df, (X_train, X_test, y_train, y_test) = \
        train_all_models(X, y, le)

    # ── 4. Save ───────────────────────────────────────────────
    save_model(best_model, scaler, le, best_name)

    # ── 5. Visualizations ─────────────────────────────────────
    print("\n===== Generating Visualizations =====")
    os.makedirs("outputs", exist_ok=True)

    plot_confusion_matrix(best_model, X_test, y_test, le,
                          save_path="outputs/confusion_matrix.png")
    plot_feature_importance(best_model, best_name,
                            save_path="outputs/feature_importance.png")
    plot_mood_clusters(df, scaler,
                       save_path="outputs/mood_clusters.png")
    plot_model_comparison(results_df,
                          save_path="outputs/model_comparison.png")

    # ── 6. Quick recommendation demo ─────────────────────────
    print("\n===== Quick Recommendation Demo =====")
    recommender = MoodMusicRecommender(df, scaler)

    for demo_mood in ["happy", "calm", "energetic"]:
        print(f"\nTop 3 songs for mood: {demo_mood.upper()}")
        recs = recommender.recommend_by_mood(demo_mood, top_n=3)
        print(recs[['title', 'artist', 'genre', 'similarity']].to_string())

    # Save one recommendation chart
    recs_happy = recommender.recommend_by_mood("happy", top_n=5)
    plot_recommendation_scores(recs_happy, "happy",
                               save_path="outputs/recommendations_happy.png")

    print("\n✅  Training complete! All outputs saved to 'outputs/'")


if __name__ == "__main__":
    main()
