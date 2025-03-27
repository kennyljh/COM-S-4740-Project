import torch
from torch.utils.data import DataLoader
from dataset import SignaturePairDataset
from model import SiameseNetwork
from utils import predict_with_threshold, plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report
import os

#setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load model
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load('siamese_signature_model.pth', map_location=device))
model.eval()

#load data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(current_dir, '..', '..', '..', 'Database', 'data', 'data'))
dataset = SignaturePairDataset(image_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

#evaluation
all_labels = []
all_preds = []
all_distances = []

with torch.no_grad():
    for img1, img2, label in dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        distances = model(img1, img2)
        preds = predict_with_threshold(distances, threshold=0.5).cpu().numpy()

        all_preds.extend(preds.flatten())
        all_labels.extend(label.cpu().numpy().flatten())
        all_distances.extend(distances.cpu().numpy().flatten())

#metrics
acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)

print(f"Accuracy: {acc * 100:.2f}%")
plot_confusion_matrix(all_labels, all_preds)
plot_roc_curve(all_labels, all_distances)

#save report
report_path = os.path.join(current_dir, 'evaluation_report.txt')
with open(report_path, 'w') as f:
    f.write("=== Final Evaluation ===\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

print(f"\nEvaluation report saved to: {report_path}")