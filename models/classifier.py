"""
models/classifier.py
Train and evaluate multiple ML models for mood classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import joblib
import os


def train_all_models(X, y, label_encoder):
    """
    Train multiple classifiers and compare them.
    Returns the best model and a comparison table.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree":    DecisionTreeClassifier(max_depth=5, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Random Forest":    RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes":      GaussianNB(),
    }

    results = []
    trained = {}

    print("\n===== Model Comparison =====")
    print(f"{'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>8}")
    print("-" * 65)

    best_acc = 0
    best_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        cv_scores = cross_val_score(model, X, y, cv=5)

        trained[name] = model
        results.append({
            "Model": name,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy":  round(test_acc,  4),
            "CV Mean":        round(cv_scores.mean(), 4),
            "CV Std":         round(cv_scores.std(),  4),
        })

        print(f"{name:<25} {train_acc:>10.2%} {test_acc:>10.2%} {cv_scores.mean():>10.2%} {cv_scores.std():>8.4f}")

        if test_acc > best_acc:
            best_acc  = test_acc
            best_name = name

    print(f"\n✅  Best model: {best_name} (test accuracy: {best_acc:.2%})")

    best_model = trained[best_name]
    print("\n===== Detailed Report for Best Model =====")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return best_model, best_name, pd.DataFrame(results), (X_train, X_test, y_train, y_test)


def save_model(model, scaler, label_encoder, model_name, path="models/saved"):
    """Save the trained model, scaler, and label encoder."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(model,         os.path.join(path, "model.pkl"))
    joblib.dump(scaler,        os.path.join(path, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(path, "label_encoder.pkl"))
    print(f"\n💾  Model saved to '{path}/'")


def load_model(path="models/saved"):
    """Load a previously saved model."""
    model         = joblib.load(os.path.join(path, "model.pkl"))
    scaler        = joblib.load(os.path.join(path, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(path, "label_encoder.pkl"))
    return model, scaler, label_encoder
