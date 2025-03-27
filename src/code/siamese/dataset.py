import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import os
from torchvision import transforms

class SignaturePairDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        valid_extensions = {'.png', '.jpg', '.jpeg'}

        self.real = []
        self.forged = []

        for root, _, files in os.walk(os.path.join(image_dir, 'real')):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.real.append(os.path.join(root, file))

        for root, _, files in os.walk(os.path.join(image_dir, 'forged')):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.forged.append(os.path.join(root, file))

        print(f"[DEBUG] Found {len(self.real)} real and {len(self.forged)} forged images")

        self.pairs = self._create_pairs()
        print(f"[DEBUG] Total pairs created: {len(self.pairs)}")

    def _create_pairs(self):
        pairs = []
        min_len = min(len(self.real), len(self.forged))
        for i in range(min_len):
            # Positive pair (same class - real)
            img1 = self.real[i]
            img2 = random.choice(self.real)
            pairs.append((img1, img2, 0))

            # Negative pair (different class)
            img1 = self.real[i]
            img2 = random.choice(self.forged)
            pairs.append((img1, img2, 1))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        try:
            img1 = Image.open(img1_path).convert('L')
            img2 = Image.open(img2_path).convert('L')
        except Exception as e:
            print(f"[WARNING] Skipped invalid image pair: {img1_path}, {img2_path}")
            return self[random.randint(0, len(self) - 1)]  # return random valid pair

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)
