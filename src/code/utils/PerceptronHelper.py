import os
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

# Reuse the same data loading/processing as SVM
from .SVMHelper import validate_image, load_data  

def extract_flattened_features(images):
    # Flatten images + basic features for perceptron
    features = []
    for img in tqdm(images, desc="Extracting Perceptron Features"):
        # Simple flattening + basic stats
        flattened = img.flatten()
        stats = [img.mean(), img.std()]
        features.append(np.concatenate([flattened, stats]))
    return np.array(features)