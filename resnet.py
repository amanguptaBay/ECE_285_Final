import torch.nn as nn
from ttt import TTT_System
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling = False, batchNorm = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.batchNorm = batchNorm
        
        # Add a shortcut connection if the number of input channels is different from the number of output channels
        if in_channels != out_channels:
            if downsampling:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
                )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):        
        out = self.conv1(x)
        if self.batchNorm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.relu(out)
        if self.batchNorm:
            out = self.bn2(out)
        out += self.shortcut(x)
        
        out = self.relu(out)
        return out

class F(nn.Module):
    def __init__(self, batchNorm = False):
        super(F, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2_x = nn.Sequential(
            ResNetBlock(64, 64, batchNorm=batchNorm),
            ResNetBlock(64, 64, batchNorm=batchNorm)
        )
        
        self.conv3_x = nn.Sequential(
            ResNetBlock(64, 128, batchNorm=batchNorm),
            ResNetBlock(128, 128, batchNorm=batchNorm)
        )
        
        self.conv4_x = nn.Sequential(
            ResNetBlock(128, 256, batchNorm=batchNorm),
            ResNetBlock(256, 256, batchNorm=batchNorm)
        )
        
        self.conv5_x = nn.Sequential(
            ResNetBlock(256, 512, batchNorm=batchNorm),
            ResNetBlock(512, 512, downsampling = True, batchNorm=batchNorm)
        )
        
        self.avgpool = nn.AvgPool2d(6)
        self.flatten = nn.Flatten()
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        
        out = self.avgpool(out)
        out = self.flatten(out)
        return out

class Out(nn.Module):
    def __init__(self, outclasses = 100):
        self.fc = nn.Linear(512, outclasses)
    def forward(self, x):
        return self.fc(x)


class ResNet(nn.Module):
    def __init__(self, batchNorm = False):
        super(ResNet, self).__init__()
        self.F = F(batchNorm=batchNorm)
        self.Out = Out(100)
        
    def forward(self, x):
        out = self.F(x)
        out = self.Out(out)
        return out
    @staticmethod
    def TTT_Implementation(outClassesForMainTask, outClassesForSelfSupervisedTask, batchNorm = True):
        return TTT_System(F(batchNorm = batchNorm), Out(outclasses = outClassesForMainTask),Out(outclasses = outClassesForSelfSupervisedTask))

