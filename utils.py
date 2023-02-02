import torch
import scipy.fftpack as fft
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Tuple


def pil_to_tensor(img: Image) -> torch.Tensor :
    return TF.pil_to_tensor(img)

def pil_to_gray(img: Image) -> np.ndarray :
    return pil_to_tensor(TF.to_grayscale(img))

def plot_tensor(img: torch.Tensor, **kwargs) -> None:
    plt.imshow(img.permute(1, 2, 0), **kwargs)

def dct2d(a : torch.Tensor)-> np.ndarray:
    return fft.dct(fft.dct(a.permute(1, 2, 0).numpy().T, norm='ortho').T, norm='ortho')

def viz_dct(dct: np.ndarray, num_coeff : int = 100) -> np.ndarray:
    return np.log(dct + np.abs(dct.min()) + 1)[:num_coeff, :num_coeff]

class train_dataset(Dataset):
    def __init__(self, path: str, tf = None) -> None:
        self.csv_path = path
        self.train_df = pd.read_csv(path)
        self.train_img_names = list(self.train_df["id"])
        self.train_img_labels = list(self.train_df["label"])

        if tf:
            self.transforms = tf
        else:
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self) -> int:
        return len(self.train_img_labels)

    def __getitem__(self, index : int) -> Tuple[torch.Tensor, torch.Tensor]:
        pil_img = Image.open(f"../dataset/train/{self.train_img_names[index]}")
        img = self.transforms(pil_img)
        label = torch.tensor(self.train_img_labels[index])
        return img, label

        



