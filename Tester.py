import torch
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import pandas as pd


from Config import Config
from Utils import Utils
from FaceData import FaceDataset
from AutoEncoder import AutoEncoder


class Tester:
    def __init__(self):
        # Data Related
        self.testset = FaceDataset(split="test")
        Config.test_loader = DataLoader(self.testset,
                                        sampler=SequentialSampler(self.testset),
                                        batch_size=200,
                                        num_workers=1)
        # Model Related
        Config.model = AutoEncoder().to(Config.device)
        load_name = "00560728"
        Config.model.load_state_dict(torch.load(os.path.join(Config.save_path, f"model_{load_name}.ckpt")))
        Config.criterion = nn.MSELoss(reduction="none")

    def infer(self):
        Config.model.eval()
        anomality = []
        with torch.no_grad():
            for image_b in tqdm(Config.test_loader):
                # Forward Pass
                image_b = image_b.to(Config.device)
                reconstructed_image = Config.model(image_b)
                # Backward Pass
                loss = Config.criterion(image_b, reconstructed_image).sum(dim=[1,2,3])
                anomality.append(loss)
        anomality = torch.cat(anomality, axis=0)
        anomality = torch.sqrt(anomality).reshape(len(self.testset), 1).cpu().numpy()
        self.write_to_csv(anomality)

    def write_to_csv(self, anomality):
        df = pd.DataFrame(anomality, columns=['score'])
        df.to_csv(os.path.join(Config.output_path, f"output_{Config.time_string}.csv"), index_label='ID')


if __name__ == "__main__":
    Utils.initialization()
    tester = Tester()
    tester.infer()


