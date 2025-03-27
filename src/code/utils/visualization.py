import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Forged', 'Real'], 
                yticklabels=['Forged', 'Real'])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

def plot_feature_space(features, labels):
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(features)
        
        plt.figure(figsize=(10, 6))
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                        alpha=0.3, label=['Forged', 'Real'][label])
        plt.title("Feature Space Projection (PCA)")
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Could not plot feature space: {e}")