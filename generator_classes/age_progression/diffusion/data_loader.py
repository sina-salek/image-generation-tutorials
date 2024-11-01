
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class UTKFaceDataset(Dataset):
    """Custom Dataset for loading UTKFace images"""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the UTKFace images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Collect all image file paths
        self.image_paths = [
            os.path.join(root_dir, filename)
            for filename in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, filename)) and filename.endswith('.jpg')
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image file path
        img_path = self.image_paths[idx]
        # Load image
        image = Image.open(img_path).convert('RGB')
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Extract labels from filename
        # Filename format: [age]_[gender]_[race]_[date&time].jpg
        filename = os.path.basename(img_path)
        sample = None
        try:
            age, gender, race, _ = filename.split('_', 3)
            age = torch.tensor(int(age), dtype=torch.long)
            gender = torch.tensor(int(gender), dtype=torch.long)
            race = torch.tensor(int(race), dtype=torch.long)
            # one-hot encode race
            race_ohe = F.one_hot(race, num_classes=5).float()

            # Prepare sample
            sample = {
                'image': image,
                'age': age,
                'gender': gender,
                'race': race_ohe
            }
        except ValueError:
            print(f'Invalid filename format: {filename}.')

        return sample
