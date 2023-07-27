import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            # Input: (N,C,H,W) = (N,3,64,64)
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),  # (N,8,64,64)
            nn.ReLU(),  # (N,8,64,64)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (N,8,32,32)
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),  # (N,16,32,32)
            nn.ReLU(),  # (N,16,32,32)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (N,16,16,16)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (N,32,16,16)
            nn.ReLU(),  # (N,32,16,16)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (N,32,8,8)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            # Input : (N,32,8,8)
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),  # (N,16,16,16)
            nn.ReLU(),  # (N,16,16,16)
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),  # (N,8,32,32)
            nn.ReLU(),  # (N,8,32,32)
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=2, padding=1),  # (N,3,64,64)
            nn.Tanh()
        )

    def forward(self, x: torch.FloatTensor):
        x = x.view(-1, 32, 8, 8)
        return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.net = nn.Sequential(Encoder(),
                                 Decoder())

    def forward(self, x):
        """
        :param x: torch.FloatTensor with shape (N,C,H,W)
        :return: torch.FloatTensor with shape (N,C,H,W)
        """
        return self.net(x)


if __name__ == "__main__":
    auto_encoder = AutoEncoder()
    encoder = Encoder()
    decoder = Decoder()
    x = torch.rand(2, 3, 64, 64)
    x = encoder(x)
    x = decoder(x)
    pass