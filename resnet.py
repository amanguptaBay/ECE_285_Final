########################################################################
# TODO: Implement the forward function for the Resnet specified        #
# above. HINT: You might need to create a helper class to              # 
# define a Resnet block and then use that block here to create         #
# the resnet layers i.e. conv2_x, conv3_x, conv4_x and conv5_x         #
########################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

import torch.nn as nn

import torch.nn as nn

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


class ResNet(nn.Module):
    def __init__(self, batchNorm = False):
        super(ResNet, self).__init__()
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
        self.fc = nn.Linear(512, 100)
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
        out = self.fc(out)
        
        return out


########################################################################
#                             END OF YOUR CODE                         #
########################################################################
