import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SignaturePairDataset
from model import SiameseNetwork
from tqdm import tqdm
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(current_dir, '..', '..', '..', 'Database', 'data', 'data'))

real_path = os.path.join(data_dir, 'real')
forged_path = os.path.join(data_dir, 'forged')

batch_size = 32
epochs = 10
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Dataset
dataset = SignaturePairDataset(image_dir=data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

real_dir = os.path.join(data_dir, 'real')
forged_path = os.path.join(data_dir, 'forged')

#Model
model = SiameseNetwork().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#Training
model.train()
for epoch in range(epochs):
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for img1, img2, label in loop:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        output = model(img1, img2)
        loss = criterion(output, label.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} - Average Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), 'siamese_signature_model.pth')
print("Model saved as 'siamese_signature_model.pth'")