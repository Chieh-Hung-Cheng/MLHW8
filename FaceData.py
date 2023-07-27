import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from Config import Config


def show_image(image_array):
    # input: np_array with shape (h, w, c)
    Image.fromarray(image_array).show()


class FaceDataset(Dataset):
    def __init__(self, split):
        self.split = split
        if split == "train":
            # train_data shape 100000, 64, 64, 3
            self.data = np.load(os.path.join(Config.data_path, 'trainingset.npy'), allow_pickle=True)
        elif split == "test":
            # test_data shape: 19636, 64, 64, 3
            self.data = np.load(os.path.join(Config.data_path, 'testingset.npy'), allow_pickle=True)
        else:
            raise ValueError("Invalid split")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # (64,64,3) to (3,64,64), np_array to FloatTensor
        image_tensor = torch.FloatTensor(self.data[item])
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = (image_tensor-255/2)/(255/2) # Remap to [-1, 1]
        return image_tensor


if __name__ == "__main__":
    trainset = FaceDataset("train")
    trainset[500]
