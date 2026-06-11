"""
recommend.py
Interactive CLI for the Mood-Based Music Recommender.
Run:  python recommend.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import load_data, preprocess
from models.recommender  import MoodMusicRecommender
from models.classifier   import load_model


BANNER = """
╔══════════════════════════════════════════════╗
║      🎵  Mood-Based Music Recommender  🎵    ║
╚══════════════════════════════════════════════╝
"""

MOODS = ["happy", "sad", "energetic", "calm", "romantic", "focused", "angry", "nostalgic"]


def print_recommendations(recs, mood):
    print(f"\n🎶  Top picks for '{mood.upper()}' mood:\n")
    print(f"  {'#':<4} {'Title':<35} {'Artist':<25} {'Genre':<15} {'Match':>6}")
    print("  " + "─" * 88)
    for i, row in recs.iterrows():
        print(f"  {i:<4} {row['title']:<35} {row['artist']:<25} {row['genre']:<15} {row['similarity']:>6.2f}")
    print()


def mode_by_mood(recommender):
    """Mode 1: user picks a mood directly."""
    print("\nAvailable moods:")
    for i, m in enumerate(MOODS, 1):
        print(f"  {i}. {m.capitalize()}")

    while True:
        choice = input("\nEnter mood number or name: ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(MOODS):
            mood = MOODS[int(choice) - 1]
            break
        elif choice in MOODS:
            mood = choice
            break
        print("  Invalid choice. Try again.")

    genres = recommender.get_available_genres(mood)
    print(f"\nAvailable genres for '{mood}': {', '.join(genres)}")
    genre = input("Filter by genre? (press Enter to skip): ").strip() or None

    top_n = input("How many recommendations? [default: 5]: ").strip()
    top_n = int(top_n) if top_n.isdigit() else 5

    recs = recommender.recommend_by_mood(mood, genre=genre, top_n=top_n)
    print_recommendations(recs, mood)


def mode_by_features(recommender):
    """Mode 2: user enters audio features manually."""
    print("\nEnter audio features (values between 0.0 – 1.0, except tempo in BPM):\n")

    def get_float(prompt, lo, hi):
        while True:
            try:
                v = float(input(f"  {prompt}: "))
                if lo <= v <= hi:
                    return v
                print(f"    Must be between {lo} and {hi}.")
            except ValueError:
                print("    Please enter a number.")

    energy       = get_float("Energy       (0–1, e.g. 0.8 = high energy)",    0, 1)
    valence      = get_float("Valence      (0–1, e.g. 0.9 = very positive)",  0, 1)
    tempo        = get_float("Tempo        (BPM, e.g. 120)",                  40, 220)
    danceability = get_float("Danceability (0–1, e.g. 0.7 = danceable)",      0, 1)
    acousticness = get_float("Acousticness (0–1, e.g. 0.1 = electric)",       0, 1)

    top_n = input("\nHow many recommendations? [default: 5]: ").strip()
    top_n = int(top_n) if top_n.isdigit() else 5

    recs, predicted_mood = recommender.recommend_by_features(
        energy, valence, tempo, danceability, acousticness, top_n=top_n
    )
    print(f"\n🧠  Predicted mood from your features: {predicted_mood.upper()}")
    print_recommendations(recs, predicted_mood)


def main():
    print(BANNER)

    # Load data & build recommender
    df = load_data("data/songs.csv")
    _, _, scaler, _ = preprocess(df)
    recommender = MoodMusicRecommender(df, scaler)

    while True:
        print("Choose a mode:")
        print("  1. Recommend by mood (pick a mood label)")
        print("  2. Recommend by audio features (enter energy, valence, etc.)")
        print("  3. Exit")

        mode = input("\nYour choice [1/2/3]: ").strip()

        if mode == "1":
            mode_by_mood(recommender)
        elif mode == "2":
            mode_by_features(recommender)
        elif mode == "3":
            print("\n👋  Goodbye!\n")
            break
        else:
            print("  Invalid choice.")

        again = input("Get more recommendations? [y/n]: ").strip().lower()
        if again != 'y':
            print("\n👋  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
