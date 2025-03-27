from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from utils.RFHelper import load_data, extract_features
from utils.visualization import plot_confusion_matrix, plot_feature_space
import numpy as np
import matplotlib.pyplot as plt
from time import time

def train_model(X_train, y_train):
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
    )
    
    print("\n=== Training Diagnostics ===")
    start_time = time()
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1)
    print(f"Cross-val scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}")
    
    # Full training
    pipeline.fit(X_train, y_train)
    print(f"\nTraining completed in {time()-start_time:.1f}s")
    
    return pipeline

def evaluate_model(model, X_test, y_test):
    print("\n=== Final Evaluation ===")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    # Load and validate data
    X, y = load_data()
    
    # Feature extraction
    X_features = extract_features(X)
    
    # Feature space visualization
    plot_feature_space(X_features, y)
    
    # Train-validation-test split
    # First split: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_features, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # Second split: 20% val, 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    print(f"\nData splits:")
    print(f"Train: {len(X_train)} samples ({len(X_train)/len(X_features):.0%})")
    print(f"Val: {len(X_val)} samples ({len(X_val)/len(X_features):.0%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(X_features):.0%})")
    
    # Train and evaluate
    model = train_model(X_train, y_train)
    
    # Validate
    val_accuracy = model.score(X_val, y_val)
    print(f"\nValidation Accuracy: {val_accuracy:.2f}")
    
    # Final test
    evaluate_model(model, X_test, y_test)

    # Feature importance
    if hasattr(model.named_steps['randomforestclassifier'], 'feature_importances_'):
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(model['randomforestclassifier'].feature_importances_)),
                model['randomforestclassifier'].feature_importances_)
        plt.title("Feature Importances")
        plt.show()