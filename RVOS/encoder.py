import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
from .vision import VGG16, ResNet34, ResNet50, ResNet101
import math
import sys

def get_skip_dims(model_name):
    if model_name == 'resnet50' or model_name == 'resnet101':
        skip_dims_in = [2048,1024,512,256,64]
    elif model_name == 'resnet34':
        skip_dims_in = [512,256,128,64,64]
    elif model_name =='vgg16':
        skip_dims_in = [512,512,256,128,64]

    return skip_dims_in

class FeatureExtractor(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self,args):
        super(FeatureExtractor,self).__init__()
        skip_dims_in = get_skip_dims(args.base_model)

        if args.base_model == 'resnet34':
            self.base = ResNet34()
            self.base.load_state_dict(models.resnet34(pretrained=True).state_dict())
        elif args.base_model == 'resnet50':
            self.base = ResNet50()
            self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())
        elif args.base_model == 'resnet101':
            self.base = ResNet101()
            self.base.load_state_dict(models.resnet101(pretrained=True).state_dict())
        elif args.base_model == 'vgg16':
            self.base = VGG16()
            self.base.load_state_dict(models.vgg16(pretrained=True).state_dict())

        else:
            raise Exception("The base model you chose is not supported !")

        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        self.padding = 0 if self.kernel_size == 1 else 1

        self.sk5 = nn.Conv2d(skip_dims_in[0],int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk4 = nn.Conv2d(skip_dims_in[1],int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk3 = nn.Conv2d(skip_dims_in[2],int(self.hidden_size/2),self.kernel_size,padding=self.padding)
        self.sk2 = nn.Conv2d(skip_dims_in[3],int(self.hidden_size/4),self.kernel_size,padding=self.padding)

        self.bn5 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn4 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn3 = nn.BatchNorm2d(int(self.hidden_size/2))
        self.bn2 = nn.BatchNorm2d(int(self.hidden_size/4))

    def forward(self, x, semseg=False, raw = False):
        x5,x4,x3,x2,x1 = self.base(x)

        x5_skip = self.bn5(self.sk5(x5))
        x4_skip = self.bn4(self.sk4(x4))
        x3_skip = self.bn3(self.sk3(x3))
        x2_skip = self.bn2(self.sk2(x2))

        if semseg:
            return x5
        elif raw:
            return x5, x4, x3, x2, x1
        else:
            #return total_feats
            del x5, x4, x3, x2, x1, x
            return x5_skip, x4_skip, x3_skip, x2_skip
