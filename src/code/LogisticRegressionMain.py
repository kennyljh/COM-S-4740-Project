from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from utils.PerceptronHelper import load_data, extract_flattened_features  # Reuse feature extraction
from utils.visualization import plot_confusion_matrix, plot_roc_curve
import numpy as np
import matplotlib.pyplot as plt
from time import time

def train_logistic_regression(X_train, y_train):
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='liblinear'
        )
    )
    
    print("\n=== Logistic Regression Training Diagnostics ===")
    start_time = time()
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Cross-val accuracy: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}")
    
    # Full training
    pipeline.fit(X_train, y_train)
    print(f"\nTraining completed in {time()-start_time:.1f}s")
    
    return pipeline

if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_data()
    X_features = extract_flattened_features(X)
    
    # 60-20-20 split
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
    print(f"Training: {len(X_train)} ({len(X_train)/len(y):.0%})")
    print(f"Validation: {len(X_val)} ({len(X_val)/len(y):.0%})")
    print(f"Test: {len(X_test)} ({len(X_test)/len(y):.0%})")
    
    # Train
    model = train_logistic_regression(X_train, y_train)
    
    # Validation evaluation
    val_pred = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    print("\n=== Validation Results ===")
    print(f"Accuracy: {accuracy_score(y_val, val_pred):.2f}")
    print(f"AUC: {roc_auc_score(y_val, val_probs):.2f}")
    print(classification_report(y_val, val_pred))
    
    # Test evaluation
    test_pred = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]
    print("\n=== Test Results ===")
    print(f"Accuracy: {accuracy_score(y_test, test_pred):.2f}")
    print(f"AUC: {roc_auc_score(y_test, test_probs):.2f}")
    print(classification_report(y_test, test_pred))
    
    # Visualizations
    plot_confusion_matrix(y_test, test_pred)
    plot_roc_curve(y_test, test_probs)