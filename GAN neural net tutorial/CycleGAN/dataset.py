from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, root_h, root_z, transform=None):
        self.root_h = root_h
        self.root_z = root_z
        self.transform = transform

        self.z_images = os.listdir(root_z)
        self.h_images = os.listdir(root_h)
        self.length_dataset = max(len(self.z_images), len(self.h_images))
        self.z_len = len(self.z_images)
        self.h_len = len(self.h_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        z_img = self.z_images[index % self.z_len]
        h_img = self.h_images[index % self.h_len]

        z_path = os.path.join(self.root_z, z_img)
        h_path = os.path.join(self.root_h, h_img)

        z_path = z_path.replace('\\\\', '/')
        h_path = h_path.replace('\\\\', '/')
        
        z_img = np.array(Image.open(z_path).convert('RGB'))
        h_img = np.array(Image.open(h_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=z_img, image0=h_img)
            z_img = augmentations['image']
            h_img = augmentations['image0']
        
        return z_img, h_img