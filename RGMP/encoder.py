import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1_p = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 
        self.res3 = resnet.layer2 
        self.res4 = resnet.layer3 
        self.res5 = resnet.layer4 

        # freeze BNs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_p):
        f = (in_f - Variable(self.mean)) / Variable(self.std)
        p = torch.unsqueeze(in_p, dim=1).float() # add channel dim

        x = self.conv1(f) + self.conv1_p(p) 
        x = self.bn1(x)
        x = self.relu(x)   
        x = self.maxpool(x)  
        r2 = self.res2(x)   
        r3 = self.res3(r2) 
        r4 = self.res4(r3) 
        r5 = self.res5(r4) 

        return r5, r4, r3, r2
