import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np

from Config import Config
from Utils import Utils
from FaceData import FaceDataset
from AutoEncoder import AutoEncoder


class Trainer:
    def __init__(self):
        # Data Related
        Config.train_loader = DataLoader(FaceDataset("train"),
                                         batch_size=Config.batch_size,
                                         num_workers=Config.num_worker,
                                         shuffle=True,
                                         pin_memory=True)
        # Model Related
        Config.model = AutoEncoder().to(Config.device)
        Config.optimizer = torch.optim.AdamW(Config.model.parameters(), lr=Config.learning_rate)
        Config.criterion = nn.MSELoss()
        # Epoch Related
        self.best_mean_loss = np.inf
        self.outer_pbar = tqdm(range(Config.epochs))
        self.outer_pbar.set_description(f"model_{Config.time_string}")

    def train_loop(self):
        for i in self.outer_pbar:
            mean_loss = self.train_once()
            self.summarize(mean_loss)
        self.save_model()

    def train_once(self):
        loss_list = []
        for image_b in Config.train_loader:
            # Forward Pass
            image_b = image_b.to(Config.device)
            reconstructed_image = Config.model(image_b)
            # Backward Pass
            Config.optimizer.zero_grad()
            loss = Config.criterion(reconstructed_image, image_b)
            loss.backward()
            Config.optimizer.step()
            # Stats
            loss_list.append(loss.item())
        return np.mean(loss_list)

    def summarize(self, mean_loss):
        if mean_loss < self.best_mean_loss:
            self.best_mean_loss = mean_loss
            self.save_model()
        self.outer_pbar.set_postfix({"train_mean_loss": f"{mean_loss:.3f}",
                                     "best_mean_loss": f"{self.best_mean_loss:.3f}"})

    def save_model(self):
        torch.save(Config.model.state_dict(), os.path.join(Config.save_path, f"model_{Config.time_string}.ckpt"))


if __name__ == "__main__":
    Utils.initialization()
    trainer = Trainer()
    trainer.train_loop()
