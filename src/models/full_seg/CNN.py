#IMPORTS
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels , out_channels, k = 3, p=0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = k, stride = 1, padding = p),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace = True),

            nn.Conv3d(out_channels, out_channels, kernel_size = k, stride = 1, padding = p),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        return self.conv(x)
    

class CNN(nn.Module):
    def __init__(self, in_channels = 5, out_channels = 1, features = (64, 128, 256, 512)):

        super(CNN, self).__init__()
        self.downs = nn.ModuleList()

        self.pool = nn.AvgPool3d(2, stride = 2)
        # down
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, k = 3, p=1))
            in_channels = feature
        
        self.FC1 = nn.Linear(512, 512)
        self.FC2 = nn.Linear(512, 128)
        self.FC3 = nn.Linear(128, out_channels)

    def forward(self, x):
        # print(x.shape)
        for down in self.downs:
            x = down(x)
            x = self.pool(x)
            # print(x.shape)
        
        x = torch.flatten(x,start_dim = 1)

        # print(x.shape)
        x = self.FC1(x)
        # print(x.shape)
        x = self.FC2(x)
        # print(x.shape)
        x = self.FC3(x)
        x = nn.Hardsigmoid()(x)
        # print(x.shape)
        return x

# M = CNN_test()
# x = torch.rand((1,5,16,16,16))
# print(M(x))

