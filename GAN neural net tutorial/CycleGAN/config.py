import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = 'dataset/train'
val_dir = 'dataset/val'
batch_size = 2
learning_rate = 2e-5
lambda_identity = 0.0
lambda_cycle = 10
num_workers = 4
epchos = 10
load_model = True
save_model = True
gen_h = "genh.pth.tar"
gen_z = "genz.pth.tar"
disc_h = "disch.pth.tar"
disc_z = "discz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)