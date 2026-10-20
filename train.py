"""Train the MLP sign language classifier.

Usage:
    python train.py                     # Train with train/val/test split
    python train.py --cross-validate    # 5-fold cross-validation
    python train.py --cross-validate --folds 10
"""

import argparse
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import config
from dataset import load_dataset, split_dataset, augment_dataset
from evaluate import evaluate_model, plot_confusion_matrix


def build_pipeline(verbose=True):
    """Build a sklearn pipeline: StandardScaler + MLP."""
    mlp = MLPClassifier(
        hidden_layer_sizes=config.HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=config.L2_ALPHA,
        learning_rate_init=config.LEARNING_RATE,
        max_iter=config.MAX_ITER,
        early_stopping=config.EARLY_STOPPING,
        validation_fraction=0.15,
        random_state=config.RANDOM_STATE,
        verbose=verbose,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ])


def train_and_evaluate(X, y, labels, target_per_class, no_augment):
    """Split first, augment ONLY the training set, evaluate on real held-out data."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    if not no_augment:
        X_train, y_train = augment_dataset(X_train, y_train,
                                           target_per_class=target_per_class)

    pipeline = build_pipeline()
    print("\nTraining MLP classifier...")
    pipeline.fit(X_train, y_train)

    print("\n--- Validation Set (held-out, NOT augmented) ---")
    evaluate_model(pipeline, X_val, y_val, labels)

    print("\n--- Test Set (held-out, NOT augmented) ---")
    evaluate_model(pipeline, X_test, y_test, labels)
    plot_confusion_matrix(pipeline, X_test, y_test, labels)

    return pipeline


def cross_validate(X, y, n_folds, target_per_class, no_augment):
    """Stratified k-fold CV. Augmentation applied PER FOLD on the training
    portion only — held-out folds stay clean, so reported metrics reflect
    real generalization."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=config.RANDOM_STATE)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        if not no_augment:
            X_tr, y_tr = augment_dataset(X_tr, y_tr,
                                         target_per_class=target_per_class)
        pipeline = build_pipeline(verbose=False)
        pipeline.fit(X_tr, y_tr)
        acc = pipeline.score(X_te, y_te)
        print(f"  Fold {fold}: {acc:.4f}")
        scores.append(acc)
    scores = np.array(scores)
    print(f"\n{n_folds}-Fold CV: mean={scores.mean():.4f} "
          f"(+/- {scores.std() * 2:.4f})")
    return scores


def fit_final_model(X, y, target_per_class, no_augment):
    """Fit one final model on ALL data (with augmentation) for inference."""
    if not no_augment:
        X, y = augment_dataset(X, y, target_per_class=target_per_class)
    pipeline = build_pipeline()
    print("\nFitting final model on all data...")
    pipeline.fit(X, y)
    return pipeline


def save_model(pipeline, path=None):
    """Save trained pipeline to disk."""
    if path is None:
        path = os.path.join(config.MODELS_DIR, "sign_classifier.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train sign language classifier")
    parser.add_argument("--cross-validate", action="store_true",
                        help="Run k-fold cross-validation")
    parser.add_argument("--folds", type=int, default=config.N_FOLDS,
                        help=f"Number of CV folds (default: {config.N_FOLDS})")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip data augmentation")
    parser.add_argument("--target-per-class", type=int, default=60,
                        help="Target samples per class after augmentation (default: 60)")
    args = parser.parse_args()

    X, y, labels = load_dataset(args.data_dir)

    if args.cross_validate:
        print(f"\n=== {args.folds}-Fold Cross-Validation "
              "(augmentation per-fold, held-out folds clean) ===")
        cross_validate(X, y, args.folds, args.target_per_class, args.no_augment)

    print("\n=== Train/Val/Test (held-out test never augmented) ===")
    train_and_evaluate(X, y, labels, args.target_per_class, args.no_augment)

    # Final model for inference: trained on ALL data + augmentation.
    print("\n=== Final inference model ===")
    pipeline = fit_final_model(X, y, args.target_per_class, args.no_augment)
    save_model(pipeline)


if __name__ == "__main__":
    main()
