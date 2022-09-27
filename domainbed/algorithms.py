# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
# from timm.models import create_model
# from timm.models.vision_transformer import _cfg
from domainbed.lib.visiontransformer import *
from domainbed.lib.cross_visiontransformer import CrossVisionTransformer
from domainbed.lib.MCT import MCVisionTransformer
from domainbed.lib.MDT import MDVisionTransformer
from domainbed.lib.DGT import DGVisionTransformer
from domainbed.lib.cvt import tiny_cvt,small_cvt
# from domainbed.lib.t2t import t2t_vit_t_14,t2t_vit_7,t2t_vit_t_24
# from domainbed.lib.t2t_utils import load_for_transfer_learning
from domainbed.lib.clip.clip import tokenize
from domainbed.lib.DIT import VisionTransformer as dit
from torch.nn.functional import interpolate
import einops
import domainbed.lib.Dino_vit as dino
import domainbed.lib.clip.clip as clip
from domainbed.lib.zipfloss import kl_loss, zipf_loss
import itertools
from prettytable import PrettyTable
import copy
import numpy as np
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from collections import defaultdict, OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)

from domainbed import queue_var # for making queue: CorrespondenceSelfCross
queue_sz = queue_var.queue_sz

ALGORITHMS = [
    'ERM',
    'MultiDomainDistillation',
    'Deit_dist',
    'DeitSmall',
    'DeitTiny',
    'CVTTiny',
    'DeitSmall_StrongTeachers',
    'DeitSmall_StrongTeachers_nodist',
    'MultiDomainDistillation_Dtokens_CE',
    'MultiDomainDistillation_Dtokens',
    'MultiDomainDistillation_Dtokens_patchmask',
    'Deit_DomInv',
    'MDT_self_dist_wCE',
    'Deit_Dino',
    'Deit_Dino_jac',
    'Deit_simple_augmix',
    'DeitBase_augmix_seperate'

]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):

    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.counte=0
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        

        # original_batch = make_grid(all_x, nrow=10, normalize=True)
        # if (self.counte < 2):

        #   save_image(original_batch, "/home/computervision1/DG_new_idea/domainbed/scripts/train_output/original_grid.png", normalize=False)

        self.counte+=1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)



class Vit_untrained_teacher_distill_features(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Vit_untrained_teacher_distill_features,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.Teachnetwork=return_backbone_network("clip",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        self.Teachnetwork.proj=None
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        self.lin_proj=nn.Linear(384,768).to("cuda")
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            list(self.network.parameters()) + list(self.lin_proj.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        # teachers_PACS=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        # teachers_VLCS=['domainbed/pretrained/VLCS/ERM/f9a56b59075f90e98ba6a746f020b111/best_val_model_testdom_[0]_0.8095.pkl','domainbed/pretrained/VLCS/ERM/59f2cefdd9db539bfa26ec5116ddaaa9/best_val_model_testdom_[1]_0.8914.pkl','domainbed/pretrained/VLCS/ERM/74752172f616ce9b3066d289c8461f54/best_val_model_testdom_[2]_0.8754.pkl','domainbed/pretrained/VLCS/ERM/241e42e44bf8761f8b90b71f1c0b13c4/best_val_model_testdom_[3]_0.8567.pkl']
        # teachers=teachers_VLCS
        # print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
        

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,_,_,dtok_feat=self.predictTrain(all_x)
        # pred_dist=torch.mean(pred_dist,1)
        dtok_feat=torch.squeeze(dtok_feat,dim=1)
        dtok_feat=self.lin_proj(dtok_feat)
        predTeacher=self.predictTeacher(self.Teachnetwork,all_x)
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd


        # changed for self distillation
        # loss+= Wd* F.kl_div(
        #     F.log_softmax(pred_dist / t, dim=1),
        #     F.log_softmax(pred/t, dim=1),
        #     reduction='sum',
        #     log_target=True
        # ) * (t * t) / pred_dist.numel()
        # print(dtok_feat.shape)
        # print(predTeacher.shape)

        # kl_loss=Wd*(F.kl_div(torch.clamp(dtok_feat, 1e-7, 1).log(),predTeacher.log(), reduction='batchmean',log_target=True))
        

        kl_loss= Wd* F.kl_div(
            F.log_softmax(dtok_feat/t , dim=1),
            F.log_softmax(predTeacher/t, dim=1),
            reduction='batchmean',
            log_target=True
        ) 
        # print(kl_loss)
        loss+=kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x,return_cls_dist_feat=True)
    def predictTeacher(self,net, x):
        return net(x)
    def predict(self, x):
        out,out_dist= self.network(x)
        return out

class Vit_untrained_teacher_distill_features_Wthce(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Vit_untrained_teacher_distill_features_Wthce,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.Teachnetwork=return_backbone_network("clip",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        self.Teachnetwork.proj=None
        self.Teach_proj=nn.Linear(768,num_classes).to("cuda")
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # self.lin_proj=nn.Linear(384,num_classes).to("cuda")
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            list(self.network.parameters())+ list(self.Teach_proj.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        # teachers_PACS=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        # teachers_VLCS=['domainbed/pretrained/VLCS/ERM/f9a56b59075f90e98ba6a746f020b111/best_val_model_testdom_[0]_0.8095.pkl','domainbed/pretrained/VLCS/ERM/59f2cefdd9db539bfa26ec5116ddaaa9/best_val_model_testdom_[1]_0.8914.pkl','domainbed/pretrained/VLCS/ERM/74752172f616ce9b3066d289c8461f54/best_val_model_testdom_[2]_0.8754.pkl','domainbed/pretrained/VLCS/ERM/241e42e44bf8761f8b90b71f1c0b13c4/best_val_model_testdom_[3]_0.8567.pkl']
        # teachers=teachers_VLCS
        # print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
        

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,pred_dist,_,_=self.predictTrain(all_x)
        # pred_dist=torch.mean(pred_dist,1)
        pred_dist=torch.squeeze(pred_dist,dim=1)
        # dtok_feat=self.lin_proj(dtok_feat)
        predTeacher=self.predictTeacher(self.Teachnetwork,all_x)
        predTeacher=self.Teach_proj(predTeacher)
        loss+= F.cross_entropy(pred,all_y)

        loss+= F.cross_entropy(predTeacher,all_y)
        t = self.temp
        Wd=self.Wd


        # changed for self distillation
        # loss+= Wd* F.kl_div(
        #     F.log_softmax(pred_dist / t, dim=1),
        #     F.log_softmax(pred/t, dim=1),
        #     reduction='sum',
        #     log_target=True
        # ) * (t * t) / pred_dist.numel()
        # print(dtok_feat.shape)
        # print(predTeacher.shape)

        # kl_loss=Wd*(F.kl_div(torch.clamp(dtok_feat, 1e-7, 1).log(),predTeacher.log(), reduction='batchmean',log_target=True))
        

        kl_loss= Wd* F.kl_div(
            F.log_softmax(pred_dist/t , dim=1),
            F.log_softmax(predTeacher/t, dim=1),
            reduction='batchmean',
            log_target=True
        ) 
        # print(kl_loss)
        loss+=kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x,return_cls_dist_feat=True)
    def predictTeacher(self,net, x):
        return net(x)
    def predict(self, x):
        out,out_dist= self.network(x)
        out_dist=torch.squeeze(out_dist,dim=1)
        return (out+out_dist)/2.0

class clip_distill_features_with_text(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(clip_distill_features_with_text,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.Teachnetwork=return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        # self.Teachnetwork.proj=None
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        self.lin_proj=nn.Linear(384,1024).to("cuda")
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            list(self.network.parameters()) + list(self.lin_proj.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        # teachers_PACS=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        # teachers_VLCS=['domainbed/pretrained/VLCS/ERM/f9a56b59075f90e98ba6a746f020b111/best_val_model_testdom_[0]_0.8095.pkl','domainbed/pretrained/VLCS/ERM/59f2cefdd9db539bfa26ec5116ddaaa9/best_val_model_testdom_[1]_0.8914.pkl','domainbed/pretrained/VLCS/ERM/74752172f616ce9b3066d289c8461f54/best_val_model_testdom_[2]_0.8754.pkl','domainbed/pretrained/VLCS/ERM/241e42e44bf8761f8b90b71f1c0b13c4/best_val_model_testdom_[3]_0.8567.pkl']
        # teachers=teachers_VLCS
        # print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
        

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,_,_,dtok_feat=self.predictTrain(all_x)
        # pred_dist=torch.mean(pred_dist,1)
        dtok_feat=torch.squeeze(dtok_feat,dim=1)
        dtok_feat=self.lin_proj(dtok_feat)
        PACS_classes=["dog","elephant","giraffe","guitar","horse","house","person"]
        VLCS_classes=["bird","car","chair","dog","person"]
        text_inputs_text = [(f"a photo of a {PACS_classes[c]}") for c in all_y]
        # print(text_inputs_text)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {VLCS_classes[c]}") for c in all_y]).to("cuda")
     
        visual_features=self.Teachnetwork.encode_image(all_x)
        Textual_features=self.Teachnetwork.encode_text(text_inputs)
        visual_features = visual_features / visual_features.norm(dim=1, keepdim=True)
        Textual_features = Textual_features / Textual_features.norm(dim=1, keepdim=True)
        conc_feat=torch.cat([visual_features,Textual_features],dim=-1)
   
     
        # predTeacher=self.predictTeacher(self.Teachnetwork,all_x)
        # print(predTeacher.shape)
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd


        # changed for self distillation
        # loss+= Wd* F.kl_div(
        #     F.log_softmax(pred_dist / t, dim=1),
        #     F.log_softmax(pred/t, dim=1),
        #     reduction='sum',
        #     log_target=True
        # ) * (t * t) / pred_dist.numel()
        # print(dtok_feat.shape)
        # print(predTeacher.shape)

        # kl_loss=Wd*(F.kl_div(torch.clamp(dtok_feat, 1e-7, 1).log(),predTeacher.log(), reduction='batchmean',log_target=True))
        

        kl_loss= Wd* F.kl_div(
            F.log_softmax(dtok_feat/t , dim=1),
            F.log_softmax(conc_feat/t, dim=1),
            reduction='batchmean',
            log_target=True
        ) 
        # print(kl_loss)
        loss+=kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x,return_cls_dist_feat=True)
    def predictTeacher(self,net, x):
        return net(x)
    def predict(self, x):
        out,out_dist= self.network(x)
        return out

class clip_full_distill_features(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(clip_full_distill_features,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.Teachnetwork=return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        # self.Teachnetwork.proj=None
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        self.lin_proj=nn.Linear(384,192).to("cuda")
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            list(self.network.parameters()) + list(self.lin_proj.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        # teachers_PACS=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        # teachers_VLCS=['domainbed/pretrained/VLCS/ERM/f9a56b59075f90e98ba6a746f020b111/best_val_model_testdom_[0]_0.8095.pkl','domainbed/pretrained/VLCS/ERM/59f2cefdd9db539bfa26ec5116ddaaa9/best_val_model_testdom_[1]_0.8914.pkl','domainbed/pretrained/VLCS/ERM/74752172f616ce9b3066d289c8461f54/best_val_model_testdom_[2]_0.8754.pkl','domainbed/pretrained/VLCS/ERM/241e42e44bf8761f8b90b71f1c0b13c4/best_val_model_testdom_[3]_0.8567.pkl']
        # teachers=teachers_VLCS
        # print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
        

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,_,_,dtok_feat=self.predictTrain(all_x)
        # pred_dist=torch.mean(pred_dist,1)
        dtok_feat=torch.squeeze(dtok_feat,dim=1)
        dtok_feat=self.lin_proj(dtok_feat)
        PACS_classes=["dog","elephant","giraffe","guitar","horse","house","person"]
        # text_inputs_text = [(f"a photo of a {PACS_classes[c]}") for c in all_y]
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {PACS_classes[c]}") for c in all_y]).to("cuda")
        visual_features,Textual_features=self.Teachnetwork(all_x,text_inputs)
     

        visual_features = visual_features / visual_features.norm(dim=1, keepdim=True)
        Textual_features = Textual_features / Textual_features.norm(dim=1, keepdim=True)
        conc_feat=torch.cat([visual_features,Textual_features],dim=-1)
        # print(conc_feat.shape)
     
        # predTeacher=self.predictTeacher(self.Teachnetwork,all_x)
        # print(predTeacher.shape)
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd


        # changed for self distillation
        # loss+= Wd* F.kl_div(
        #     F.log_softmax(pred_dist / t, dim=1),
        #     F.log_softmax(pred/t, dim=1),
        #     reduction='sum',
        #     log_target=True
        # ) * (t * t) / pred_dist.numel()
        # print(dtok_feat.shape)
        # print(predTeacher.shape)

        # kl_loss=Wd*(F.kl_div(torch.clamp(dtok_feat, 1e-7, 1).log(),predTeacher.log(), reduction='batchmean',log_target=True))
        

        kl_loss= Wd* F.kl_div(
            F.log_softmax(dtok_feat/t , dim=1),
            F.log_softmax(conc_feat/t, dim=1),
            reduction='batchmean',
            log_target=True
        ) 
        # print(kl_loss)
        loss+=kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x,return_cls_dist_feat=True)
    def predictTeacher(self,net, x):
        return net(x)
    def predict(self, x):
        out,out_dist= self.network(x)
        return out

class clip_with_text(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(clip_with_text,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.network=return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            list(self.network.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.Class_names=queue_var.Class_names
        self.cnt=0
       
        

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        batch_images = make_grid(all_x, nrow=7, normalize=True)
        if(self.cnt==0):
            save_image(batch_images, "./domainbed/batchimg.png",normalize=False)
        text_inputs_text = [(f"a photo of a {self.Class_names[c]}") for c in all_y]
        # print(text_inputs_text)
        # exit()
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")

        # normalized features
        if self.cnt<2000:
            with torch.no_grad():
                image_features = self.network.encode_image(all_x)
        else:
            image_features = self.network.encode_image(all_x)
            
        with torch.no_grad():
            text_features = self.network.encode_text(text_inputs)
            # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        # image_features = image_features @ self.network.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        

        # cosine similarity as logits
        logit_scale = self.network.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
   
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        print(logits_per_text.shape)

        labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        print(loss_i)
        print(loss_t)
        loss = (loss_i + loss_t)/2

        self.cnt+=1
        self.optimizer.zero_grad()
        # loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
        
    def predictTrain(self,net, x):
        return net(x)
    def predict(self, x):
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
        with torch.no_grad():
            image_features = self.network.encode_image(x)
            text_features = self.network.encode_text(text_inputs)
            # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
            # image_features = image_features @ self.network.visual.proj

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        

            # cosine similarity as logits
            logit_scale = self.network.logit_scale.exp()

            logits_per_image = logit_scale * image_features @ text_features.t()
            prob=F.softmax(logits_per_image,dim=1)
            # print(prob.shape)
            # print(prob[0])
            prob=torch.max(prob,dim=1)
            # print(prob)
          
        return logits_per_image

class domain_aware_clip_with_text(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(domain_aware_clip_with_text,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.network=return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            list(self.network.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=queue_var.Class_names
        self.dom_names=queue_var.environments.copy()
        self.dom_name=queue_var.environments.copy()
        
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del self.dom_name[test_envs[-1]]  # Deleting test environment models
        assert len(self.dom_name)==num_domains ,"Required number of teacher pretrained models not given"
        print(self.dom_names)
        

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # print(len(minibatches))
        label_chunks=torch.chunk(all_y,chunks=self.num_domains,dim=0)
        text_inputs_text =[[(f"a photo of a {self.Class_names[y]} of {self.dom_name[i]}") for y in label_chunks[i]] for i in range(self.num_domains)]
        text_inputs_text=list(itertools.chain.from_iterable(text_inputs_text))

        

        # text_inputs = torch.cat([clip.tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        # normalized features
        image_features = self.network.encode_image(all_x)
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(c) for c in text_inputs_text]).to("cuda")
            text_features = self.network.encode_text(text_inputs)
            text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.network.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        

        # cosine similarity as logits
        logit_scale = self.network.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
   
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
      

        labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t)/2


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
        
    def predictTrain(self,net, x):
        return net(x)
    def predict(self, x,env=None):
        text_inputs = torch.cat([clip.tokenize(f"a photo of  of a {c} of {self.dom_names[env]} ") for c in self.Class_names]).to("cuda")
        # print([(f"a photo of a {c} {self.dom_names[env]}") for c in self.Class_names])
        with torch.no_grad():
            image_features = self.network.encode_image(x)
            text_features = self.network.encode_text(text_inputs)
            text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
            image_features = image_features @ self.network.visual.proj

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        

            # cosine similarity as logits
            logit_scale = self.network.logit_scale.exp()

            logits_per_image = logit_scale * image_features @ text_features.t()
          
        return logits_per_image

class ERM_clip_weighted_text_conc(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_weighted_text_conc, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        self.featurizer_orig = return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)

        self.classifier = networks.Classifier(
            1280,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=queue_var.Class_names
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        image_features = self.featurizer.encode_image(all_x)
        text_features = self.featurizer.encode_text(text_inputs)
        
        with torch.no_grad():
            text_inputs_zero_sht = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            text_features_zero_shot = self.featurizer.encode_text(text_inputs_zero_sht)
            image_feature_orig=self.featurizer_orig.encode_image(all_x)
            vis_text_proj = image_feature_orig @ self.featurizer_orig.visual.proj
            
            image_features_ze = vis_text_proj / vis_text_proj.norm(dim=1, keepdim=True)
            text_features_zero_shot = text_features_zero_shot / text_features_zero_shot.norm(dim=1, keepdim=True)
            logit_scale = self.featurizer_orig.logit_scale.exp()
        
            logits_per_image = logit_scale * image_features_ze @ text_features_zero_shot.t()
            prob=F.softmax(logits_per_image,dim=1)
            vals=prob[torch.arange(len(all_x)), all_y].unsqueeze(1)
  
            # prob=torch.max(prob,dim=1)
            # vals=prob.values.unsqueeze(1)
            # print(vals.shape)
        text_features=vals*text_features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        conc_feat=torch.cat([image_features,text_features],dim=1)
        loss=F.cross_entropy(self.classifier(conc_feat), all_y)
        
        # print(conc_feat.shape)
        # if(self.cnt<2500):
        #     with torch.no_grad():
        #         feat=self.featurizer(all_x)
        #     loss=F.cross_entropy(self.classifier(feat), all_y)
        # else:
        #     loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        text_inputs = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
        
        image_features = self.featurizer_orig.encode_image(x)
        text_features = self.featurizer_orig.encode_text(text_inputs)
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer_orig.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        # cosine similarity as logits
        logit_scale = self.featurizer_orig.logit_scale.exp()
        

        logits_per_image = logit_scale * image_features @ text_features.t()
        prob=F.softmax(logits_per_image,dim=1)
        prob=torch.max(prob,dim=1)
        vals=prob.values.unsqueeze(1)
        indices=prob.indices
        text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in indices]).to("cuda")
        image_features = self.featurizer.encode_image(x)
        text_features = self.featurizer.encode_text(text_inputs)
        text_features=vals*text_features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        conc_feat=torch.cat([image_features,text_features],dim=1)
        outs=self.classifier(conc_feat)
        return outs


class domain_prompt_clip(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(domain_prompt_clip,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.network=return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        printNetworkParams(self.network)
        self.lin_proj=nn.Linear(512,768).to("cuda")
        self.classifier = networks.Classifier(
            768,
            num_classes,
            self.hparams['nonlinear_classifier']).to("cuda")
        self.optimizer = torch.optim.AdamW(  
            list(self.network.parameters()) + list(self.lin_proj.parameters())+ list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        VLCS_classes=["bird","car","chair","dog","person"]
        PACS_classes=["dog","elephant","giraffe","guitar","horse","house","person"]
        if num_classes==7:
            self.Class_names=PACS_classes
        elif num_classes==5:
            self.Class_names=VLCS_classes
        self.dom_names=['art painting', 'cartoon', 'photo','sketch']
        self.dom_name=['art painting', 'cartoon', 'photo','sketch']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del self.dom_name[test_envs[-1]]  # Deleting test environment models
        assert len(self.dom_name)==num_domains ,"Required number of teacher pretrained models not given"
       
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # print(len(minibatches))
        label_chunks=torch.chunk(all_y,chunks=self.num_domains,dim=0)
        text_inputs_text =[[(f"{self.dom_name[i]}") for y in label_chunks[i]] for i in range(self.num_domains)]
        text_inputs_text=list(itertools.chain.from_iterable(text_inputs_text))

        

        # text_inputs = torch.cat([clip.tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        # normalized features
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(c) for c in text_inputs_text]).to("cuda")
            text_features = self.network.encode_text(text_inputs)
            text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)]@ self.network.text_projection
        
        text_features=self.lin_proj(text_features)
        # print(text_features.shape)
        if self.cnt<2500:
            with torch.no_grad():
        
                image_features = self.network.encode_image(all_x,text_feat=text_features)
        else:
            image_features = self.network.encode_image(all_x,text_feat=text_features)
        pred=self.classifier(image_features)
        # image_features = image_features @ self.network.visual.proj

        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        

        # # cosine similarity as logits
        # logit_scale = self.network.logit_scale.exp()

        # logits_per_image = logit_scale * image_features @ text_features.t()
   
        # logits_per_text = logits_per_image.t()

        # # shape = [global_batch_size, global_batch_size]
      

        # labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        # loss_i = F.cross_entropy(logits_per_image, labels)
        # loss_t = F.cross_entropy(logits_per_text, labels)
        # loss = (loss_i + loss_t)/2

        loss = F.cross_entropy(pred, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}
        
    def predictTrain(self,net, x):
        return net(x)
    def predict(self, x,env=None):
        text_inputs = clip.tokenize(f"{self.dom_names[env]}").to("cuda")
        text_inputs = text_inputs.expand(x.shape[0], -1) 
        # print([(f"a photo of a {c} {self.dom_names[env]}") for c in self.Class_names])
        with torch.no_grad():
            
            text_features = self.network.encode_text(text_inputs)
            
        text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)]
        text_features=self.lin_proj(text_features)
        # print(text_features.shape)
        image_features = self.network.encode_image(x,text_feat=text_features)
        pred=self.classifier(image_features)
        return pred


class domain_aware_clip_with_text_prompting(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(domain_aware_clip_with_text_prompting,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.network=return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            list(self.network.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        VLCS_classes=["bird","car","chair","dog","person"]
        PACS_classes=["dog","elephant","giraffe","guitar","horse","house","person"]
        if num_classes==7:
            self.Class_names=PACS_classes
        elif num_classes==5:
            self.Class_names=VLCS_classes
        self.dom_names=['art painting', 'cartoon', 'photo','sketch']
        self.dom_name=['art painting', 'cartoon', 'photo','sketch']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del self.dom_name[test_envs[-1]]  # Deleting test environment models
        assert len(self.dom_name)==num_domains ,"Required number of teacher pretrained models not given"
       
        

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # print(len(minibatches))
        label_chunks=torch.chunk(all_y,chunks=self.num_domains,dim=0)
        text_inputs_text =[[(f"a photo of a {self.Class_names[y]} {self.dom_name[i]}") for y in label_chunks[i]] for i in range(self.num_domains)]
        text_inputs_text=list(itertools.chain.from_iterable(text_inputs_text))

        text_inputs = torch.cat([clip.tokenize(c) for c in text_inputs_text]).to("cuda")
        print(text_inputs.shape)
        
        # text_inputs = torch.cat([clip.tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        # normalized features
        with torch.no_grad():
            image_features = self.network.encode_image(all_x)
            text_features = self.network.encode_text(text_inputs)
            print(text_features.shape)
        text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        print(text_features.shape)
        image_features = image_features @ self.network.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        

        # cosine similarity as logits
        logit_scale = self.network.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
   
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
      

        labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t)/2


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
        
    def predictTrain(self,net, x):
        return net(x)
    def predict(self, x,env=None):
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c} ") for c in self.Class_names]).to("cuda")
        # print([(f"a photo of a {c} {self.dom_names[env]}") for c in self.Class_names])
        with torch.no_grad():
            image_features = self.network.encode_image(x)
            text_features = self.network.encode_text(text_inputs)
            text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
            image_features = image_features @ self.network.visual.proj

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        

            # cosine similarity as logits
            logit_scale = self.network.logit_scale.exp()

            logits_per_image = logit_scale * image_features @ text_features.t()
          
        return logits_per_image




class CLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model =  return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print('Set self.clip_model.parameters.reguires_grad = False!')

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512  # 
        self.Class_names=queue_var.Class_names
        classnames = [name.replace('_', ' ') for name in self.Class_names]
        self.prompt = torch.cat([clip.tokenize(f'a photo of a {ppt}') for ppt in classnames]).to(self.device)
        
    def update(self, minibatches, unlabeled=None):
        return {'loss': 0}
    
    def predict(self, x):
        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)
     

# rename to DPL (Domain Prompt Learning)
class DPLCLIP(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams, sentence_prompt=True):
        super(DPLCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)

        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        self.Class_names=queue_var.Class_names
        if sentence_prompt:
            print('Using sentence_prompt in DPLCLIP...')
            classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.Class_names]
        else:
            classnames = [name.replace('_', ' ') for name in self.Class_names]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
        # prompts:  ['X X X X X X X X dog.', 'X X X X X X X X elephant.' ...]
        
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)
        
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS
        # torch.Size([7, 68, self.EMBEDDING_DIM]), 68 := 77 - num_domain_tokens_tokens - 2.
        # [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.
        
        self.network = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.network.apply(init_weights)
        self.num_domains=num_domains
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
            
    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]

        ##########Edited
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        label_chunks=torch.chunk(all_y,chunks=self.num_domains,dim=0)
        all_y =[[len(self.Class_names)*i+y for y in label_chunks[i]] for i in range(self.num_domains)]
        all_y=torch.tensor(list(itertools.chain.from_iterable(all_y))).cuda().long()
        # print(all_y)
        # all_y= torch.cat([data[1].cuda().long() for data in minibatches]) ##########Edited
        #  encode image for each domain.
        image_features = [self.clip_model.encode_image(x) @ self.clip_model.visual.proj for x in all_x]
        
        #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] -> [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
        domain_features = [self.network(feature) for feature in image_features]
        image_features = torch.cat(image_features)
        #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
        mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]

        #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
        _mean_domain_features = [feature.repeat_interleave(len(self.Class_names), dim=0) for feature in mean_domain_features]
        
        #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
        # text_features = [self._get_text_features(feature) for feature in _mean_domain_features]
        text_features = torch.cat([self._get_text_features(feature) for feature in _mean_domain_features])
            
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()
        # print(logits_per_image.shape)
        # print(all_y)
        loss = F.cross_entropy(logits_per_image, all_y)
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}


    def _get_text_features(self, domain_feature, coop=False):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
        domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
        
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)
        
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        
        #  mapping domain_features to text_features.
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection      
        return text_features

    def predict(self, x):
        image_feature = self.clip_model.encode_image(x) @ self.clip_model.visual.proj
        
        domain_feature = self.network(image_feature)
        mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.Class_names), dim=0)
        text_feature = self._get_text_features(mean_domain_feature)
        
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return self.clip_model.logit_scale.exp() * image_feature @ text_feature.t()


class ERM_clip_WTC_DPL(DPLCLIP):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_WTC_DPL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = return_backbone_network("clip_full",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)

        self.classifier = networks.Classifier(
            1280,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters())+list(self.network.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # self.optimizer = torch.optim.SGD(
        #     self.network.parameters(),
        #     lr=self.hparams["lr"],
        #     momentum=self.hparams["momentum"]
        # )
        self.Class_names=queue_var.Class_names
        # print(self.Class_names)
        self.cnt=0


    def update(self, minibatches, unlabeled=None):
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        # text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        image_features = self.featurizer.encode_image(torch.cat(all_x,dim=0))
        # text_features_lrn = self.clip_model.encode_text(text_inputs)
        
        #######
        label_chunks=torch.chunk(all_y,chunks=self.num_domains,dim=0)
        all_y_d =[[len(self.Class_names)*i+y for y in label_chunks[i]] for i in range(self.num_domains)]
        all_y_d=torch.tensor(list(itertools.chain.from_iterable(all_y_d))).cuda().long()
        with torch.no_grad():
        #  encode image for each domain.
            vis_text_proj = [self.clip_model.encode_image(x) @ self.clip_model.visual.proj for x in all_x]
     
        
        #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] -> [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
        domain_features = [self.network(feature) for feature in vis_text_proj]
        vis_text_proj = torch.cat(vis_text_proj)
        #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
        mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]

        #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
        _mean_domain_features = [feature.repeat_interleave(len(self.Class_names), dim=0) for feature in mean_domain_features]
        
        #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
        # text_features = [self._get_text_features(feature) for feature in _mean_domain_features]
        text_features = torch.cat([self._get_text_features(feature) for feature in _mean_domain_features])
            
        image_feature_orig = vis_text_proj / vis_text_proj.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_feature_orig @ text_features_norm.t()

        prob=F.softmax(logits_per_image,dim=1)
        vals=prob[torch.arange(len(vis_text_proj)), all_y_d].unsqueeze(1)
        loss = F.cross_entropy(logits_per_image, all_y_d)
        ######
        text_features=torch.index_select(text_features, 0, all_y_d)

        # text_features=vals*text_features
        
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        conc_feat=torch.cat([image_features,text_features],dim=1)
        loss+=F.cross_entropy(self.classifier(conc_feat), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cnt+=1
        return {'loss': loss.item()}


    def predict(self, x):

        image_feature = self.clip_model.encode_image(x) @ self.clip_model.visual.proj
        
        domain_feature = self.network(image_feature)

        mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.Class_names), dim=0)
        text_feature = self._get_text_features(mean_domain_feature)
        
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)
        logits_per_image=self.clip_model.logit_scale.exp() * image_feature @ text_feature_norm.t()

        prob=F.softmax(logits_per_image,dim=1)
        prob=torch.max(prob,dim=1)
        vals=prob.values.unsqueeze(1)
        indices=prob.indices
        # text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in indices]).to("cuda")
        image_features = self.featurizer.encode_image(x)

        text_feature=torch.index_select(text_feature, 0, indices)
        # text_feature=vals*text_feature

        conc_feat=torch.cat([image_features,text_feature],dim=1)
        outs=self.classifier(conc_feat)
        return outs

class Deit_DI_tokening_momentum(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_DI_tokening_momentum,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.num_classes=num_classes
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
    
        attn_sep_mask=self.hparams["attn_sep_mask"]
        # self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        self.network=dit(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,in_chans=3,add_di_token=True,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Wd=self.hparams['Wd']
        self.class_features=np.zeros((num_classes,384))
        self.cnt=0
        self.embed_dim=384
        self.delta=self.hparams['delta']


    def update(self, minibatches, unlabeled=None):
        class_features=Variable(torch.Tensor(self.class_features)).to('cuda')
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,dtok_features=self.predictTrain(all_x)
        loss+= F.cross_entropy(pred,all_y)

        #class wise average across domains for the current batch
        dtok_features=F.softmax(dtok_features, dim=1)
        dtok_feat=torch.chunk(dtok_features,chunks=self.num_domains,dim=0)
        dom_aver_feat=torch.zeros(self.num_classes,self.embed_dim).to('cuda')
        class_presense=torch.zeros(self.num_classes)
        for dom in range (self.num_domains):
            cls_feat=torch.zeros(self.num_classes,self.embed_dim).to('cuda').index_add_(0,minibatches[dom][1],dtok_feat[dom])
            class_counts=torch.bincount(minibatches[dom][1],minlength=self.num_classes)
            class_presense[class_counts>0]+=1
            class_counts[class_counts==0] = 1
            cls_feat=torch.div(cls_feat,class_counts.reshape(-1,1))
            dom_aver_feat=dom_aver_feat+cls_feat

        class_presense[class_presense==0.0]=1.0
        dom_aver_feat=torch.div(dom_aver_feat,class_presense.to('cuda').reshape(-1,1))

        if(self.cnt>0):
            class_features=self.delta*dom_aver_feat+(1-self.delta)*class_features
        else:
            class_features=dom_aver_feat
        
        # class_features=torch.div(class_features,torch.sum(class_features,dim=1,keepdim=True))
        sum=torch.sum(class_features,dim=1,keepdim=True)
        sum[sum==0.0]=1.0
        class_features=torch.div(class_features,sum)
        Wd=self.Wd
        domInvFeatures=torch.index_select(class_features,0,all_y)

        # Clamp mixture distribution to avoid exploding KL divergence
        kl_loss=Wd*(F.kl_div( dtok_features.log(),torch.clamp(domInvFeatures, 1e-7, 1), reduction='batchmean'))
        # print(kl_loss)
        loss+=kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.class_features=class_features.detach().to('cpu').numpy()
        self.cnt+=1
        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)

    def predict(self, x):
        out,out_dist= self.network(x)
        return out



class DI_tokening(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,backbone="DeitSmall"):
        super(DI_tokening,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # network_deit=deit_small_patch16_224(pretrained=True) 
        # network_deit.head = nn.Linear(384, num_classes)
    
        # attn_sep_mask=self.hparams["attn_sep_mask"]
        # # self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        # self.network=dit(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,in_chans=3,add_di_token=True,attn_sep_mask=attn_sep_mask)
        # self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        # printNetworkParams(self.network)
        self.network=return_backbone_network(backbone,num_classes,self.hparams)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Wd=self.hparams['Wd']
        self.cnt=0
        self.num_class_select=self.hparams['num_class_select']


    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        n=self.num_class_select
        d=self.num_domains
        all_y=einops.rearrange(all_y,'(n d s)->(d s n)',n=n ,d=d)
        all_x=einops.rearrange(all_x,'(n d s) C H W->(d s n) C H W ',n=n ,d=d)

        
        batch_images = make_grid(all_x, nrow=7, normalize=True)
        if(self.cnt==0):
            save_image(batch_images, "./domainbed/batchimg.png",normalize=False)
        loss=0
        pred,dtok_feat=self.predictTrain(all_x)
        loss+= F.cross_entropy(pred,all_y)
    
        Wd=self.Wd

        dtok_feat=torch.chunk(dtok_feat,chunks=self.num_domains,dim=0)
        domft0,domft1,domft2 = F.softmax(dtok_feat[0], dim=1), F.softmax(dtok_feat[1], dim=1), F.softmax(dtok_feat[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1)
        loss += Wd * (F.kl_div(domft0.log(),domInvFtAvg , reduction='batchmean') +
                        F.kl_div(domft1.log(),domInvFtAvg,  reduction='batchmean') +
                        F.kl_div(domft2.log(),domInvFtAvg,  reduction='batchmean')) / 3.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)

    def predict(self, x):
        out,out_dist= self.network(x)
        return out


class ERM_ViT(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,backbone="DeitSmall"):
        super(ERM_ViT,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        #for clip changed
        self.network=return_backbone_network(backbone,num_classes,self.hparams,add_di_token=False)
        # self.network.head()
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.cnt=0
        self.num_class_select=self.hparams['num_class_select']


    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # n=self.num_class_select
        # d=self.num_domains
        # all_y=einops.rearrange(all_y,'(n d s)->(d s n)',n=n ,d=d)
        # all_x=einops.rearrange(all_x,'(n d s) C H W->(d s n) C H W ',n=n ,d=d)
      
        batch_images = make_grid(all_x, nrow=7, normalize=True)
        if(self.cnt==0):
            save_image(batch_images, "./domainbed/batchimg.png",normalize=False)
        loss=0
        pred=self.predictTrain(all_x)

        loss+= F.cross_entropy(pred,all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)

    def predict(self, x):
        out= self.network(x)
        return out

class DI_tokening_vit(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,backbone="DeitSmall"):
        super(DI_tokening_vit,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # network_deit=deit_small_patch16_224(pretrained=True) 
        # network_deit.head = nn.Linear(384, num_classes)
    
        # attn_sep_mask=self.hparams["attn_sep_mask"]
        # # self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        # self.network=dit(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,in_chans=3,add_di_token=True,attn_sep_mask=attn_sep_mask)
        # self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        # printNetworkParams(self.network)
        self.network=return_backbone_network(backbone,num_classes,self.hparams)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Wd=self.hparams['Wd']
        self.cnt=0
        self.num_class_select=self.hparams['num_class_select']


    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        n=self.num_class_select
        d=self.num_domains
        all_y=einops.rearrange(all_y,'(n d s)->(d s n)',n=n ,d=d)
        all_x=einops.rearrange(all_x,'(n d s) C H W->(d s n) C H W ',n=n ,d=d)

        
        batch_images = make_grid(all_x, nrow=7, normalize=True)
        if(self.cnt==0):
            save_image(batch_images, "./domainbed/batchimg.png",normalize=False)
        loss=0
        pred,dtok_feat=self.predictTrain(all_x)
        loss+= F.cross_entropy(pred,all_y)
    
        Wd=self.Wd

        dtok_feat=torch.chunk(dtok_feat,chunks=self.num_domains,dim=0)
        domft0,domft1,domft2 = F.softmax(dtok_feat[0], dim=1), F.softmax(dtok_feat[1], dim=1), F.softmax(dtok_feat[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1).log()
        loss += Wd * (F.kl_div(domInvFtAvg ,domft0, reduction='batchmean') +
                        F.kl_div(domInvFtAvg,domft1,  reduction='batchmean') +
                        F.kl_div(domInvFtAvg,domft2,  reduction='batchmean')) / 3.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)

    def predict(self, x):
        out,out_dist= self.network(x)
        return out




def load_model(fname):
    dump = torch.load(fname)
    algorithm_class = get_algorithm_class(dump["args"]["algorithm"])
    algorithm = algorithm_class(
        dump["model_input_shape"],
        dump["model_num_classes"],
        dump["model_num_domains"],
        dump["model_hparams"])
    algorithm.load_state_dict(dump["model_dict"])
    return algorithm

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def printNetworkParams(net):
    # print("network1====",net)
    # count_parameters(net)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("pytorch_total_params:",pytorch_total_params)
    print("pytorch_total_trainable_params:",pytorch_total_trainable_params)

def similarityCE(pred,predTarget):
    total_loss = 0
    n_loss_terms = 0
    temp=0.5
    pred=pred/temp
    predTarget=F.softmax(predTarget/temp, dim=-1)
    for iq, q in enumerate(predTarget):
        for v in range(len(pred)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = torch.sum(-q * F.log_softmax(pred[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms 
    return total_loss

def UpdateClsTrainQueues(minibatches):
    train_queues=queue_var.train_queues
    nclass=len(train_queues)
    ndomains=len(train_queues[0])
    for id_c in range(nclass): # loop over classes
        for id_d in range(ndomains): # loop over domains
            mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
            # indices of those egs from domain id_d, whose class label is id_c
            label_tensor=minibatches[id_d][1][mb_ids] # labels
            if mb_ids.size(0)==0:
                #print('class has no element')
                continue
            data_tensor=minibatches[id_d][0][mb_ids] # data
            data_tensor = data_tensor.detach()
            
            # update queue for this class and this domain
            current_queue = train_queues[id_c][id_d]
            current_queue = torch.cat((current_queue, data_tensor), 0)
            current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
            train_queues[id_c][id_d] = current_queue
            # all_labels+=label_tensor
    queue_var.train_queues=train_queues
    return train_queues

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
    elif(model_name=="DinoSmall"):
        model = dino.__dict__['vit_base'](patch_size=16, num_classes=num_classes,distilled=distilled,num_dist_token=num_dist_token)
        # for p in model.parameters():
        #     p.requires_grad = False
        # model.eval()
        model.to(device)
        pretrained_weights="domainbed/pretrained/dino/dino_deitbase16_pretrain.pth"
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    return model

def get_per_sample_jaccard(pred, target):
    jac = 0
    object_count = 0
    for mask_idx in torch.unique(target):
        if mask_idx in [0, 255]:  # ignore index
            continue
        cur_mask = target == mask_idx
        intersection = (cur_mask * pred) * (cur_mask != 255)  # handle void labels
        intersection = torch.sum(intersection, dim=[1, 2])  # handle void labels
        union = ((cur_mask + pred) > 0) * (cur_mask != 255)
        union = torch.sum(union, dim=[1, 2])
        jac_all = intersection / union
        jac += jac_all.max().item()
        object_count += 1
    return jac / object_count


def compute_jaccard(attentions_Batch,target_batch):
    # make the image divisible by the patch size
    # attentions_Batch:B,nh,196
    # target_batch:B,nh,196
    w_featmap = 14
    h_featmap = 14
    nh=attentions_Batch.shape[1]
    # th_attn_all=[]
    threshold=0.75
    patch_size=16
    jacc_val=0

    for b in range(attentions_Batch.shape[0]):
        attentions=attentions_Batch[b,:,:]
        
        # we keep only the output patch attention
        # print("attentions.shape:",attentions.shape)

        attentions = attentions.reshape(nh, -1)

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cum_val = torch.cumsum(val, dim=1)
        th_attn = cum_val > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]

        target=target_batch[b,:,:]
        target = target.reshape(nh, -1)

        # we keep only a certain percentage of the mass
        tval, tidx = torch.sort(target)
        tval /= torch.sum(tval, dim=1, keepdim=True)
        tcum_val = torch.cumsum(tval, dim=1)
        th_attn_tar = tcum_val > (1 - threshold)
        tidx2 = torch.argsort(tidx)
        for head in range(nh):
            th_attn_tar[head] = th_attn_tar[head][tidx2[head]]
        th_attn_tar = th_attn_tar.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn_tar = interpolate(th_attn_tar.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
        jacc_val+=get_per_sample_jaccard(th_attn,th_attn_tar)
        # th_attn_all.append(th_attn)
    jacc_val=jacc_val/(attentions_Batch.shape[0]*1.0)

    return jacc_val



def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def get_jaccard_loss_from_attention(pred_attn,target_attn,threshold=0.75,N_patch=14):
    '''
    pred_attn: predicted attentions [B, H, N,N]  Eg: B,6,197,197
    target_attn: target attentions [B, H, N,N]
    '''
    pred_attn=pred_attn[:,:,0,1:] #cls token attention
    target_attn = target_attn[:, :, 0, 1:] #cls token attention

    pred_attn = torch.mean(pred_attn,dim=1).reshape(-1,1,N_patch, N_patch).float() # mean over heads 

    target_attn=torch.mean(target_attn,dim=1)
    val, idx = torch.sort(target_attn)
    B=target_attn.shape[0]
    val /= torch.sum(val, dim=1, keepdim=True)
    cum_val = torch.cumsum(val, dim=1)
    th_attn = cum_val > (1 - threshold)
    idx2 = torch.argsort(idx)
    for sample in range(B):
        th_attn[sample] = th_attn[sample][idx2[sample]]
    th_attn = th_attn.reshape(B,N_patch,N_patch).long()


    return jaccard_loss(th_attn,pred_attn)

def return_backbone_network(network_name,num_classes,hparams,add_di_token=False,distilled=False,num_dist_token=0):
    print(network_name)
    if(network_name=="DeitSmall"):
        if not (add_di_token):
            network = deit_small_patch16_224(pretrained=True)
            network.head = nn.Linear(384, num_classes)
            return network
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
    
        attn_sep_mask=hparams["attn_sep_mask"]
        # self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        network=dit(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,in_chans=3,add_di_token=add_di_token,attn_sep_mask=attn_sep_mask)
        network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(network)
        return network
    elif(network_name=="DeitBase"):
        if not (add_di_token):
            network = deit_base_patch16_224(pretrained=True)
            network.head = nn.Linear(768, num_classes)
            return network
        network_deit=deit_base_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(768, num_classes)
        attn_sep_mask=hparams["attn_sep_mask"]
        network=dit(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=768, depth=12, num_heads=12,in_chans=3,add_di_token=add_di_token,attn_sep_mask=attn_sep_mask)
        network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(network)
        return network
    elif(network_name=="CVTSmall"):
        network=small_cvt(pretrained=True,add_di_tok=add_di_token) 
        network.head = nn.Linear(384, num_classes)
        attn_sep_mask=hparams["attn_sep_mask"]
        printNetworkParams(network)
        return network
    elif(network_name=="CVTTiny"):
        network=tiny_cvt(pretrained=True,add_di_tok=add_di_token) 
        network.head = nn.Linear(384, num_classes)
        attn_sep_mask=hparams["attn_sep_mask"]
        printNetworkParams(network)
        return network
    elif(network_name=="T2T7"):
        attn_sep_mask=hparams["attn_sep_mask"]
        network=t2t_vit_7(pretrained=True,add_di_tok=add_di_token,attn_sep_mask=attn_sep_mask)
        network.head = nn.Linear(256, num_classes) # for t2t-7-> 256 for t2t24->512
        printNetworkParams(network)
        return network
    elif(network_name=="T2T24"):
        attn_sep_mask=hparams["attn_sep_mask"]
        network=t2t_vit_t_24(pretrained=True,add_di_tok=add_di_token,attn_sep_mask=attn_sep_mask)
        network.head = nn.Linear(512, num_classes) # for t2t-7-> 256 for t2t24->512
        printNetworkParams(network)
        return network
    elif(network_name=="DinoSmall"):
        network=load_dino(network_name,num_classes,distilled=distilled,num_dist_token=num_dist_token)
        printNetworkParams(network)
        return network
    elif(network_name=="clip"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/16', device)
        model=model.float()
        network=model.visual
        network.proj=nn.Parameter(0.03608439182435161 * torch.randn(768, num_classes))
        # network.proj=None
        # classifier=nn.Linear(768, num_classes)
        
        return network
    elif(network_name=="clip_full"):
        #change
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/16', device)
        model=model.float()
        network=model
        
        # network.proj=nn.Parameter(0.03608439182435161 * torch.randn(768, num_classes))
        # network.proj=None
        # classifier=nn.Linear(768, num_classes)
        
        return network
    
    
