import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class GC(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1), padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw), padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw), padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1), padding=(int(kh/2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        
        return x

class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.convFS1(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr       
        m = s + F.upsample(pm, scale_factor=self.scale_factor, mode='bilinear')
        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        
        return m


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        mdim = 256
        self.GC = GC(4096, mdim) 
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.RF4 = Refine(1024, mdim) 
        self.RF3 = Refine(512, mdim) 
        self.RF2 = Refine(256, mdim)

        self.pred5 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred4 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred3 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r5, x5, r4, r3, r2):
        x = torch.cat((r5, x5), dim=1)

        x = self.GC(x) 
        r = self.convG1(F.relu(x))
        r = self.convG2(F.relu(r))
        m5 = x + r            
        m4 = self.RF4(r4, m5) 
        m3 = self.RF3(r3, m4)
        m2 = self.RF2(r2, m3)

        p2 = self.pred2(F.relu(m2))
        p3 = self.pred3(F.relu(m3))
        p4 = self.pred4(F.relu(m4))
        p5 = self.pred5(F.relu(m5))
        
        p = F.upsample(p2, scale_factor=4, mode='bilinear')
        
        return p, p2, p3, p4, p5
