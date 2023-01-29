# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from gradinit import GradInitWrapper
from domainbed.lib import wide_resnet
import copy
from domainbed.lib.visiontransformer import *
from domainbed.lib.cvt import small_cvt
import domainbed.lib.clip.clip as clip
import domainbed.lib.Dino_vit as dino
# from domainbed.lib.CvT import CvT

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['weight_init']=="ImageNet":
            if hparams['backbone']=="Resnet18":
                self.network = torchvision.models.resnet18(pretrained=True)
                self.n_outputs = 512
            else:
                self.network = torchvision.models.resnet50(pretrained=True)
                self.n_outputs = 2048
        elif hparams['weight_init']=="MoCoV2":
            if hparams['backbone']=="Resnet18":
                self.network = torchvision.models.resnet18()
                self.n_outputs = 512
            else:
                path="domainbed/pretrained/Resnet50/mocov2_resnet50_8xb32-coslr-200e_in1k_20220225-89e03af4.pth"
                self.network = load_resnet_from_path(path, num_classes=7)
                self.n_outputs = 2048
        elif hparams['weight_init']=="BYOL":
            if hparams['backbone']=="Resnet18":
                self.network = torchvision.models.resnet18()
                self.n_outputs = 512
            else:
                path="domainbed/pretrained/Resnet50/byol_resnet50_8xb32-accum16-coslr-300e_in1k_20220225-a0daa54a.pth"
                self.network = load_resnet_from_path(path, num_classes=7)
                self.n_outputs = 2048
        elif hparams['weight_init']=="kaiming_normal":
            if hparams['backbone']=="Resnet18":
                self.network = torchvision.models.resnet18()
                self.n_outputs = 512
            else:
                self.network = torchvision.models.resnet50()
                self.n_outputs = 2048
        elif hparams['weight_init']=="xavier_uniform":
            if hparams['backbone']=="Resnet18":
                self.network = torchvision.models.resnet18()
                self.n_outputs = 512
            else:
                self.network = torchvision.models.resnet50()
                self.n_outputs = 2048
            self.apply(self._init_weights_xavier_uniform)

        elif hparams['weight_init']=="trunc_normal":
            if hparams['backbone']=="Resnet18":
                self.network = torchvision.models.resnet18()
                self.n_outputs = 512
            else:
                self.network = torchvision.models.resnet50()
                self.n_outputs = 2048
            self.apply(self._init_weights_trunc_normal)
        elif hparams['weight_init']=="gradinit":
            if hparams['backbone']=="Resnet18":
                self.network = torchvision.models.resnet18()
                self.n_outputs = 512
            else:
                self.network = torchvision.models.resnet50()
                self.n_outputs = 2048
        
            ginit = GradInitWrapper(self.network)
            ginit.detach() 
        elif hparams['weight_init']=="uniform":
            if hparams['backbone']=="Resnet18":
                self.network = torchvision.models.resnet18()
                self.n_outputs = 512
            else:
                self.network = torchvision.models.resnet50()
                self.n_outputs = 2048
            self.apply(self._init_weights_uniform)
        # self.network = remove_batch_norm_from_resnet(self.network)
            
        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
       

    def _init_weights_xavier_uniform(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.xavier_uniform_(module.bias)
    def _init_weights_uniform(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.uniform_(module.bias)

    def _init_weights_trunc_normal(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.trunc_normal_(module.weight,std=.02)
            if module.bias is not None:
                torch.nn.init.trunc_normal_(module.bias,std=.02)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)  

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class ViT(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams,num_classes):
        super(ViT, self).__init__()
        if hparams['weight_init']=="ImageNet":
            if hparams['backbone']=="DeitSmall":
                self.network = deit_small_patch16_224(pretrained=True)
                self.network.head = nn.Linear(384, num_classes)
                self.n_outputs = 384
            elif hparams['backbone']=="CVTSmall":
                self.network = small_cvt(pretrained=True)
                self.network.head = nn.Linear(384, num_classes)
            elif hparams['backbone']=="DeitBase":
                self.network = deit_base_patch16_224(pretrained=True)
                self.network.head = nn.Linear(768, num_classes)
                self.n_outputs = 768
            elif hparams['backbone']=="DeitBase_dist":
                self.network = deit_base_distilled_patch16_224(pretrained=True)
                self.network.head = nn.Linear(768, num_classes)
                self.network.head_dist = nn.Linear(768, num_classes)
                self.n_outputs = 768
            
            else:
                raise NotImplementedError
        elif hparams['weight_init']=="ImageNet21k":
            if hparams['backbone']=="VitBase":
                self.network = vit_base_patch16_224_in21k(pretrained=True)
                self.network.head = nn.Linear(768, num_classes)
            else:
                raise NotImplementedError
        elif hparams['weight_init']=="Dino":
            if hparams['backbone']=="DeitSmall":
                self.network = load_dino("DinoSmall",num_classes=num_classes)
                self.n_outputs = 384
                # self.network.head = nn.Linear(384, num_classes)
            elif hparams['backbone']=="DeitBase":
                self.network = load_dino("DinoBase",num_classes=num_classes)
                self.n_outputs = 768
                # self.network.head = nn.Linear(768, num_classes)
            else:
                raise NotImplementedError
        elif hparams['weight_init']=="clip":
            if hparams['backbone']=="DeitBase":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load('ViT-B/16', device)
                # model=model.float()
                self.network=model.visual.float()
                
                self.network.proj=nn.Parameter(0.03608439182435161 * torch.randn(768, num_classes))
                self.n_outputs = 768
                # self.network.proj==None
            else:
                raise NotImplementedError       
        elif hparams['weight_init']=="clip_full":
            if hparams['backbone']=="DeitBase":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load('ViT-B/16', device)
                # model=model.float()
                self.network=model.float()
                self.n_outputs = 768
                # self.network.proj=nn.Parameter(0.03608439182435161 * torch.randn(768, num_classes))
                # self.network.proj==None
            elif hparams['backbone']=="Resnet50":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load('RN50', device)
                # model=model.float()
                self.network=model.float()
                self.n_outputs = 1024
                # self.network.proj=nn.Parameter(0.03608439182435161 * torch.randn(768, num_classes))
                # self.network.proj==None
            else:
                raise NotImplementedError     
        elif hparams['weight_init']=="clip_scratch":
            if hparams['backbone']=="DeitBase":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load('ViT-B/16', device,scratch=True)
                # model=model.float()
                self.network=model.float()
                
                # self.network.proj=nn.Parameter(0.03608439182435161 * torch.randn(768, num_classes))
                # self.network.proj==None
            else:
                raise NotImplementedError   
        
        elif hparams['weight_init']=="Random":
            if hparams['backbone']=="DeitSmall":
                self.network = deit_small_patch16_224(pretrained=False)
                self.network.head = nn.Linear(384, num_classes)
            elif hparams['backbone']=="CVTSmall":
                self.network = small_cvt(pretrained=False)
                self.network.head = nn.Linear(384, num_classes)
            elif hparams['backbone']=="DeitBase":
                self.network = deit_base_patch16_224(pretrained=False)
                self.network.head = nn.Linear(768, num_classes)
            else:
                raise NotImplementedError
        elif hparams['weight_init']=="xavier_uniform":
            if hparams['backbone']=="DeitSmall":
                self.network = deit_small_patch16_224(pretrained=False,weight_init='xavier')
                self.n_outputs = 384
                self.network.head = Classifier(self.n_outputs, num_classes,init=hparams['weight_init'])
                
            elif hparams['backbone']=="CVTSmall":
                self.network = small_cvt(pretrained=False,init="xavier")
                self.network.head = nn.Linear(384, num_classes)
            else:
                raise NotImplementedError
            self.apply(self._init_weights_xavier_uniform)
        elif hparams['weight_init']=="gradinit":
            if hparams['backbone']=="DeitSmall":
                self.network = deit_small_patch16_224(pretrained=False)
                self.n_outputs = 384
                self.network.head = Classifier(self.n_outputs, num_classes,init=hparams['weight_init'])
            elif hparams['backbone']=="CVTSmall":
                self.network = small_cvt(pretrained=False)
                self.network.head = nn.Linear(384, num_classes)
            else:
                raise NotImplementedError
            ginit = GradInitWrapper(self.network)
            ginit.detach() 
        elif hparams['weight_init']=="trunc_normal":
            if hparams['backbone']=="DeitSmall":
                self.network = deit_small_patch16_224(pretrained=False)
                self.n_outputs = 384
                self.network.head = Classifier(self.n_outputs, num_classes,init=hparams['weight_init'])
            elif hparams['backbone']=="CVTSmall":
                self.network = small_cvt(pretrained=False,init="xavier")
                self.network.head = nn.Linear(384, num_classes)
            else:
                raise NotImplementedError
            self.apply(self._init_weights_trunc_normal)
        elif hparams['weight_init']=="kaiming_normal":
            if hparams['backbone']=="DeitSmall":
                self.network = deit_small_patch16_224(pretrained=False)
                self.n_outputs = 384
                self.network.head = Classifier(self.n_outputs, num_classes,init=hparams['weight_init'])
            elif hparams['backbone']=="CVTSmall":
                self.network = small_cvt(pretrained=False,init="xavier")
                self.network.head = nn.Linear(384, num_classes)
            else:
                raise NotImplementedError
            self.apply(self._init_weights_kaiming_normal)
        # self.network = remove_batch_norm_from_resnet(self.network)

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def _init_weights_xavier_uniform(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
           
    def _init_weights_uniform(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.uniform_(module.bias)

    def _init_weights_trunc_normal(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight,std=.02)
            if module.bias is not None:
                torch.nn.init.trunc_normal_(module.bias,std=.02)
    def _init_weights_kaiming_normal(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        # return self.dropout(self.network(x))
        return self.network(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if( hparams["backbone"]=="DeitSmall" or hparams["backbone"]=="CVTSmall"):
        print("not supported for this algorithm")
        raise NotImplementedError
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False,init=None):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        lin= torch.nn.Linear(in_features, out_features)
        if init=="xavier_uniform":
            torch.nn.init.xavier_uniform_(lin.weight)
            lin.bias.data.fill_(0.01)
        elif init=="trunc_normal":
            torch.nn.init.trunc_normal_(lin.weight,std=.02)
            lin.bias.data.fill_(0.01)
        elif init=="gradinit":
            ginit = GradInitWrapper(lin)
            ginit.detach() 
        elif init=="uniform":
            torch.nn.init.uniform_(lin.weight)
            lin.bias.data.fill_(0.01)
        elif init=="kaiming_normal":
            torch.nn.init.kaiming_normal_(lin.weight)
            lin.bias.data.fill_(0.01)
        return lin


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

def load_dino(model_name,num_classes=0,distilled=False,num_dist_token=0):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if(model_name=="DinoSmall"):
    # build model
        model = dino.__dict__['vit_small'](patch_size=16, num_classes=num_classes,distilled=distilled,num_dist_token=num_dist_token)
        # for p in model.parameters():
        #     p.requires_grad = False
        # model.eval()
        model.to(device)
        pretrained_weights="domainbed/pretrained/dino/dino_deitsmall16_pretrain.pth"
    elif(model_name=="DinoBase"):
        model = dino.__dict__['vit_base'](patch_size=16, num_classes=num_classes,distilled=distilled,num_dist_token=num_dist_token)
        # for p in model.parameters():
        #     p.requires_grad = False
        # model.eval()
        model.to(device)
        pretrained_weights="domainbed/pretrained/dino/dino_vitbase16_pretrain.pth"
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    return model

def load_resnet_from_path(path, num_classes=0):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torchvision.models.resnet50(pretrained=True)
    # for p in model.parameters():
    #     p.requires_grad = False
    # model.eval()
    model.to(device)
    state_dict = torch.load(path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(path, msg))
    return model