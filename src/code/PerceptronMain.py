from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from utils.PerceptronHelper import load_data, extract_flattened_features
from utils.visualization import plot_confusion_matrix, plot_feature_space
import numpy as np
import matplotlib.pyplot as plt
from time import time

def train_perceptron(X_train, y_train):
    pipeline = make_pipeline(
        StandardScaler(),
        Perceptron(
            penalty='l2',           # Regularization
            alpha=0.0001,           # Learning rate
            class_weight='balanced',
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    )
    
    print("\n=== Perceptron Training Diagnostics ===")
    start_time = time()
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
    print(f"Cross-val scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}")
    
    # Full training
    pipeline.fit(X_train, y_train)
    print(f"\nTraining completed in {time()-start_time:.1f}s")
    
    return pipeline

if __name__ == "__main__":
    # Load and validate data (same as SVM)
    X, y = load_data()
    
    # Feature extraction - simpler approach for perceptron
    X_features = extract_flattened_features(X)
    
    # Feature space visualization
    plot_feature_space(X_features, y)
    
    # 60-20-20 Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    
    print("\n=== Data Splits ===")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(y):.0%})")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/len(y):.0%})")
    print(f"Test samples: {len(X_test)} ({len(X_test)/len(y):.0%})")
    
    # Train and evaluate
    model = train_perceptron(X_train, y_train)
    
    # Validation evaluation
    val_pred = model.predict(X_val)
    print("\n=== Validation Results ===")
    print(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.2f}")
    print(classification_report(y_val, val_pred))
    
    # Final test evaluation
    test_pred = model.predict(X_test)
    print("\n=== Final Evaluation ===")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.2f}")
    print(classification_report(y_test, test_pred))
    plot_confusion_matrix(y_test, test_pred)