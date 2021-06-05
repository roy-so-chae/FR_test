def inverse_transform(images):
    return (images+1.)/2.

def initialize_netG(net,input_frame_nums=4):
    print("Loading weights from PyTorch ResNet101")
    pretrained_dict = torch.load(os.path.join('./models', 'resnet101_pytorch.pth'))
    model_dict = net.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    weight = pretrained_dict['conv1.weight']
    pretrained_dict['conv1.weight'] = torch.cat([weight]*input_frame_nums,1)
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(model_dict)


def initialize_netD(net):
    print("Loading weights from PyTorch GoogleNet")
    pretrained_dict = torch.load(os.path.join('./models', 'inception_v3_google_pytorch.pth'))
    model_dict = net.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('fc' not in k) and ('AuxLogits' not in k)}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(model_dict)
