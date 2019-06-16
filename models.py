import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable
import torchvision as tv
import torch

class OmniglotModel(nn.Module):
    def __init__(self, num_classes, C=64):
        super(OmniglotModel, self).__init__()
        self.conv =  nn.ModuleList()
        self.totalConvLayers = 4
        self.num_classes = num_classes
        #first layer
        self.conv.append(nn.Conv2d(3,C,3,stride=2,padding=1))
        for _ in range(1, self.totalConvLayers):
            self.conv.append(nn.Conv2d(C, C, 3, stride=2, padding=1))
        
        self.bn = nn.ModuleList()
        for _ in range(self.totalConvLayers):
            self.bn.append(nn.BatchNorm2d(C))
            
        #output should be a 64*2*2
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        for i in range(self.totalConvLayers):
            x = F.relu(self.bn[i](self.conv[i](x)))
        x = x.view(-1, len(x)).transpose(0,1)
        return self.classifier(x)
        
        
        
        
        
        
        
        