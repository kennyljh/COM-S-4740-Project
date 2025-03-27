import os
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

def validate_image(img):
    """Check for common image issues"""
    if img.mean() < 10 or img.max() < 50:
        return "low_contrast"
    if np.percentile(img, 99) > 250:
        return "overexposed"
    if np.isclose(img.std(), 0):
        return "uniform"
    return None

def load_data():
    data_dir = Path("../data")
    images, labels = [], []
    valid_extensions = {".jpg", ".jpeg", ".png"}
    issues = {"low_contrast": 0, "overexposed": 0, "uniform": 0}

    for label, subdir in enumerate(["real", "forged"]):
        dir_path = data_dir / subdir
        
        for img_file in dir_path.rglob("*"):
            if img_file.suffix.lower() not in valid_extensions:
                continue
                
            try:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (128, 128))
                
                # Image validation
                issue = validate_image(img)
                if issue:
                    issues[issue] += 1
                    continue
                    
                # Preprocessing
                img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                img = cv2.filter2D(img, -1, kernel)
                
                images.append(img)
                labels.append(label)
                
            except Exception as e:
                print(f"Skipped {img_file.relative_to(data_dir)}: {str(e)}")

    print("\n=== Data Quality Report ===")
    print(f"Loaded {len(images)} valid images")
    print(f"Rejected images - Low contrast: {issues['low_contrast']}, Overexposed: {issues['overexposed']}, Uniform: {issues['uniform']}")
    return np.array(images), np.array(labels)

def extract_hog_features(images):
    hog_features = []
    for img in tqdm(images, desc="Extracting HOG Features"):
        features = hog(
            img,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            block_norm='L2-Hys'
        )
        hog_features.append(features)
    return np.array(hog_features)