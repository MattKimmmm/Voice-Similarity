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
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
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
        # print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"After CNN shape: {x.shape}")
        x = x.view(x.size()[0], -1)
        # (4, 16, 320) -> (4, 5120)
        # print(f"After flattening shape: {x.shape}")
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2
    
# Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=10.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
# Dropout layers at inputs of CNN and FCNN
class Siamese_dropout(nn.Module):
    def __init__(self):
        super(Siamese_dropout, self).__init__()
    
        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Dropout1d(0.2),
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # Input (4, 1, 320)
            # L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1] = 320
            # Output (4, 16, 320)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 16, 320) -> (4, 16, 160)

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # (4, 16, 160) -> (4, 32, 160)
            nn.ReLU(inplace=True)
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(5120, 1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 640),
            nn.ReLU(inplace=True),

            nn.Linear(640, 320),
            nn.ReLU(inplace=True),

            nn.Linear(320, 160)
        )

    def forward_once(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"After CNN shape: {x.shape}")
        x = x.view(x.size()[0], -1)
        # (4, 16, 320) -> (4, 5120)
        # print(f"After flattening shape: {x.shape}")
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2

# Dropout layers at Hidden Layers of CNN and FCNN
class Siamese_dropout_hidden(nn.Module):
    def __init__(self):
        super(Siamese_dropout_hidden, self).__init__()
    
        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # Input (4, 1, 320)
            # L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1] = 320
            # Output (4, 16, 320)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 16, 320) -> (4, 16, 160)
            nn.Dropout1d(0.3),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # (4, 16, 160) -> (4, 32, 160)
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.3)
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(5120, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1280, 640),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(640, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(320, 160)
        )

    def forward_once(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"After CNN shape: {x.shape}")
        x = x.view(x.size()[0], -1)
        # (4, 16, 320) -> (4, 5120)
        # print(f"After flattening shape: {x.shape}")
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2
    

class Siamese_st16(nn.Module):
    def __init__(self):
        super(Siamese_st16, self).__init__()
    
        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=16, stride=16),
            # Input (4, 1, 320)
            # L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1] = 320
            # Output (4, 8, 20)
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
            # (4, 8, 20) -> (4, 16, 18)
            nn.ReLU(inplace=True)
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(288, 144),
            nn.ReLU(inplace=True),

            nn.Linear(144, 72),
            nn.ReLU(inplace=True),

            nn.Linear(72, 36)
        )

    def forward_once(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"After CNN shape: {x.shape}")
        x = x.view(x.size()[0], -1)
        # (4, 16, 18) -> (4, 288)
        # print(f"After flattening shape: {x.shape}")
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2
    
class Siamese_fc(nn.Module):
    def __init__(self):
        super(Siamese_fc, self).__init__()

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(320, 640),
            nn.ReLU(inplace=True),

            nn.Linear(640, 320),
            nn.ReLU(inplace=True),

            nn.Linear(320, 160)
        )

    def forward_once(self, x):
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2
    

class Siamese_st8(nn.Module):
    def __init__(self):
        super(Siamese_st8, self).__init__()
    
        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=16, stride=8),
            # Input (4, 1, 320)
            # L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1] = 320
            # Output (4, 8, 39)
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, stride=1),
            # (4, 8, 20) -> (4, 16, 36)
            nn.ReLU(inplace=True)
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(576, 288),
            nn.ReLU(inplace=True),

            nn.Linear(288, 144),
            nn.ReLU(inplace=True),

            nn.Linear(144, 72)
        )

    def forward_once(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"After CNN shape: {x.shape}")
        x = x.view(x.size()[0], -1)
        # (4, 16, 18) -> (4, 288)
        # print(f"After flattening shape: {x.shape}")
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2
    

class Siamese_Conv(nn.Module):
    def __init__(self):
        super(Siamese_Conv, self).__init__()
    
        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # Input (4, 1, 320)
            # L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1] = 320
            # Output (4, 16, 320)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 16, 320) -> (4, 16, 160)

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # (4, 16, 160) -> (4, 32, 160)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 32, 160) -> (4, 32, 80)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
            # (4, 32, 80) -> (4, 64, 80)
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
        # print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"After CNN shape: {x.shape}")
        x = x.view(x.size()[0], -1)
        # (4, 16, 320) -> (4, 5120)
        # print(f"After flattening shape: {x.shape}")
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2
    
class Siamese_Conv_fc(nn.Module):
    def __init__(self):
        super(Siamese_Conv_fc, self).__init__()
    
        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # Input (4, 1, 320)
            # L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1] = 320
            # Output (4, 16, 320)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 16, 320) -> (4, 16, 160)

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # (4, 16, 160) -> (4, 32, 160)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 32, 160) -> (4, 32, 80)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
            # (4, 32, 80) -> (4, 64, 80)
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(5120, 2560),
            nn.ReLU(inplace=True),

            nn.Linear(2560, 1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 640),
            nn.ReLU(inplace=True),

            nn.Linear(640, 320),
            nn.ReLU(inplace=True),

            nn.Linear(320, 160)
        )

    def forward_once(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"After CNN shape: {x.shape}")
        x = x.view(x.size()[0], -1)
        # (4, 16, 320) -> (4, 5120)
        # print(f"After flattening shape: {x.shape}")
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2
    
# Last part of the ablation study, dropout layer added to the siamese_conv
class Siamese_Conv1_dropout(nn.Module):
    def __init__(self):
        super(Siamese_Conv1_dropout, self).__init__()
    
        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Dropout1d(0.2),
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # Input (4, 1, 320)
            # L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1] = 320
            # Output (4, 16, 320)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 16, 320) -> (4, 16, 160)

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # (4, 16, 160) -> (4, 32, 160)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 32, 160) -> (4, 32, 80)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
            # (4, 32, 80) -> (4, 64, 80)
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Dropout1d(0.2),
            nn.Linear(5120, 1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 640),
            nn.ReLU(inplace=True),

            nn.Linear(640, 320),
            nn.ReLU(inplace=True),

            nn.Linear(320, 160)
        )

    def forward_once(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"After CNN shape: {x.shape}")
        x = x.view(x.size()[0], -1)
        # (4, 16, 320) -> (4, 5120)
        # print(f"After flattening shape: {x.shape}")
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2

# one additional conv layer
class Siamese_Conv2_dropout(nn.Module):
    def __init__(self):
        super(Siamese_Conv2_dropout, self).__init__()
    
        # Convolutional Layer
        self.cnn = nn.Sequential(
            nn.Dropout1d(0.2),
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # Input (4, 1, 320)
            # L_out = [(L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1] = 320
            # Output (4, 16, 320)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 16, 320) -> (4, 16, 160)

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # (4, 16, 160) -> (4, 32, 160)
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),
            # (4, 32, 160) -> (4, 32, 80)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # (4, 32, 80) -> (4, 64, 80)
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Dropout1d(0.2),
            nn.Linear(5120, 1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 640),
            nn.ReLU(inplace=True),

            nn.Linear(640, 320),
            nn.ReLU(inplace=True),

            nn.Linear(320, 160)
        )

    def forward_once(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"After CNN shape: {x.shape}")
        x = x.view(x.size()[0], -1)
        # (4, 16, 320) -> (4, 5120)
        # print(f"After flattening shape: {x.shape}")
        x = self.fc(x)
        return x
    def forward(self, intput1, intput2):
        # print("forward_SiameseNetwork")
        output1 = self.forward_once(intput1)
        output2 = self.forward_once(intput2)
        return output1, output2