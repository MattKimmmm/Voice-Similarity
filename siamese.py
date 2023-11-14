import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
    
        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # Input (4, 1, 320)
            # L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1] = 320
            # Output (4, 16, 320)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2),
            # (4, 16, 320) -> (4, 16, 160)

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # (4, 16, 160) -> (4, 32, 160)
            nn.ReLU(inplace=True)
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(5120, 1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 640),
            nn.ReLU(inplace=True),

            nn.Linear(640, 320),
            nn.ReLU(inplace=True),

            nn.Linear(320, 160)
        )

    def forward_once(self, x):
        print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        # (4, 16, 320) -> (4, 5120)
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2
    
# Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
