#IMPORTS
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels , out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1), #, bias = False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace = True),

            nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        return self.conv(x)
    

class UNET(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, features = (32, 64, 128)):#, 256)): #(16, 32, 64, 128))

        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size = 2, stride = 2)

        # down
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        #up
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size = 2, stride = 2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        #bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        #
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size = 1)
    
    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups) , 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:

                x = x.flatten(0,1)
                x = TF.resize(x, skip_connection.shape[3:])
                x = x.swapaxes(1,2)
                skip_connection = skip_connection.swapaxes(2,3)
                x = TF.resize(x, skip_connection.shape[3:])
                x = x.swapaxes(1,2)
                skip_connection = skip_connection.swapaxes(2,3)
                x = x.view(1,*x.shape)

            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
