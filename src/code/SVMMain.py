from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from utils.SVMHelper import load_data, extract_hog_features
from utils.visualization import plot_confusion_matrix, plot_feature_space
import numpy as np
import matplotlib.pyplot as plt
from time import time

def train_svm(X_train, y_train):
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(
            kernel='poly',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=42
        )
    )
    
    print("\n=== SVM Training Diagnostics ===")
    start_time = time()
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
    print(f"Cross-val scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}")
    
    # Full training
    pipeline.fit(X_train, y_train)
    print(f"\nTraining completed in {time()-start_time:.1f}s")
    print(f"Number of support vectors: {len(pipeline.named_steps['svc'].support_vectors_)}")
    
    return pipeline

if __name__ == "__main__":
    # Load and validate data
    X, y = load_data()
    
    # Feature extraction (HOG only for SVM)
    X_hog = extract_hog_features(X)
    
    # Feature space visualization
    plot_feature_space(X_hog, y)
    
    # Train-test split
    # First split: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_hog, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )
    
    # Second split: 20% val, 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,  # Splits 40% into 20% val and 20% test
        random_state=42,
        stratify=y_temp
    )
    
    print("\n=== Data Splits ===")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(X_hog):.0%})")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X_hog):.0%})")
    print(f"Test samples: {len(X_test)} ({len(X_test)/len(X_hog):.0%})")
    
    # Train and evaluate
    svm_model = train_svm(X_train, y_train)
    
    # Validation evaluation
    val_pred = svm_model.predict(X_val)
    print("\n=== Validation Results ===")
    print(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.2f}")
    print(classification_report(y_val, val_pred))

    # Final test evaluation
    test_pred = svm_model.predict(X_test)
    print("\n=== Final Evaluation ===")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.2f}")
    print(classification_report(y_test, test_pred))
    plot_confusion_matrix(y_test, test_pred)