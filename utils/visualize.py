"""
utils/visualize.py
Visualizations: confusion matrix, feature importance, mood clusters, similarity scores.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


MOOD_COLORS = {
    "happy":     "#FAC775",
    "sad":       "#85B7EB",
    "energetic": "#F0997B",
    "calm":      "#5DCAA5",
    "romantic":  "#ED93B1",
    "focused":   "#AFA9EC",
    "angry":     "#F09595",
    "nostalgic": "#97C459",
}


def plot_confusion_matrix(model, X_test, y_test, label_encoder, save_path="confusion_matrix.png"):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    labels = label_encoder.classes_

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_xlabel("Predicted Mood", fontsize=12)
    ax.set_ylabel("Actual Mood",    fontsize=12)
    ax.set_title("Confusion Matrix — Mood Classification", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(model, model_name, save_path="feature_importance.png"):
    feature_cols = ['energy', 'valence', 'tempo', 'danceability', 'acousticness', 'popularity']

    if not hasattr(model, 'feature_importances_'):
        print(f"  {model_name} does not expose feature_importances_ — skipping.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        [feature_cols[i] for i in indices],
        importances[indices],
        color=["#AFA9EC", "#5DCAA5", "#FAC775", "#F0997B", "#85B7EB", "#ED93B1"][:len(indices)],
        edgecolor='white', height=0.6
    )
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, importances[indices]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mood_clusters(df, scaler, save_path="mood_clusters.png"):
    """PCA scatter plot showing how songs cluster by mood."""
    feature_cols = ['energy', 'valence', 'tempo', 'danceability', 'acousticness', 'popularity']
    X_scaled = scaler.transform(df[feature_cols])

    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    moods = df['mood'].values

    for mood in sorted(set(moods)):
        mask = moods == mood
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=MOOD_COLORS.get(mood, '#888'),
                   label=mood.capitalize(), s=80, alpha=0.85, edgecolors='white', linewidth=0.5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=11)
    ax.set_title("Song Mood Clusters (PCA)", fontsize=14, fontweight='bold')
    ax.legend(title="Mood", bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_recommendation_scores(recommendations, mood, save_path="recommendations.png"):
    """Bar chart of similarity scores for recommended songs."""
    fig, ax = plt.subplots(figsize=(9, 4))
    labels  = [f"{r['title']} — {r['artist']}" for _, r in recommendations.iterrows()]
    scores  = recommendations['similarity'].values
    color   = MOOD_COLORS.get(mood, '#AFA9EC')

    bars = ax.barh(labels, scores, color=color, edgecolor='white', height=0.55)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_title(f"Top Recommendations for '{mood.capitalize()}' mood", fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, scores):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results_df, save_path="model_comparison.png"):
    """Grouped bar chart comparing models."""
    models  = results_df['Model']
    x       = np.arange(len(models))
    width   = 0.28

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, results_df['Train Accuracy'], width, label='Train',  color='#AFA9EC', edgecolor='white')
    ax.bar(x,          results_df['Test Accuracy'],  width, label='Test',   color='#5DCAA5', edgecolor='white')
    ax.bar(x + width,  results_df['CV Mean'],        width, label='CV Mean',color='#FAC775', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison", fontsize=13, fontweight='bold')
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
