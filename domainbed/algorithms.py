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

class MultiDomainDistillation(Algorithm):
    """
    MultiDomainDistillation Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MultiDomainDistillation, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # load_model_path='domainbed/pretrained/single_train_models/DeitSmall/820147d8f3bc2473e6b839f2e4fb0f2e/best_val_model_valdom_2_0.9940.pkl' #should not be the test model
        # deit_trained_dgbed=load_model(load_model_path)
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        self.network=MCVisionTransformer(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_cls_emb=num_domains)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/pretrained/single_train_models/ERM/c52ccf0a2ff8fcb27e5f0c637efb5f90/best_val_model_valdom_0_0.9511.pkl','domainbed/pretrained/single_train_models/ERM/c62da0be82f89501d09e0a0e8fd65200/best_val_model_valdom_1_0.9744.pkl','domainbed/pretrained/single_train_models/ERM/cde183b2a672bd049e697054ef75e988/best_val_model_valdom_2_0.9940.pkl','domainbed/pretrained/single_train_models/ERM/6d29d7ebbbc85f0d744e924bb602d5ad/best_val_model_valdom_3_0.9720.pkl']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del teachers[test_envs[-1]]  # Deleting test environment models
        assert len(teachers)==num_domains ,"Required number of teacher pretrained models not given"
        self.Teachnetwork=[load_model(fname).to("cuda") for fname in teachers]
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss=0
        predictions=self.predictTrain(all_x)
        loss+= F.cross_entropy(torch.mean(predictions,dim=1), all_y)

        predictions=torch.chunk(predictions,self.num_domains,dim=0)
        t = self.temp
        Wd=self.Wd
        for dom in range(self.num_domains):
            loss+= F.cross_entropy(predictions[dom][:,dom], minibatches[dom][1])
            loss+= Wd* F.kl_div(
                F.log_softmax(predictions[dom][:,dom] / t, dim=1),
                F.log_softmax(self.predictTeacher(self.Teachnetwork[dom],minibatches[dom][0]) / t, dim=1),
                reduction='sum',
                log_target=True
            ) * (t * t) / predictions[dom][:,dom].numel()
    

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predict(self, x):
        return self.network(x)[:,2]
        return torch.sum(self.network(x),1)
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)


class MultiDomainDistillation_NoKL(Algorithm):
    """
    MultiDomainDistillation Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MultiDomainDistillation_NoKL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        load_model_path='domainbed/pretrained/single_train_models/DeitSmall/820147d8f3bc2473e6b839f2e4fb0f2e/best_val_model_valdom_2_0.9940.pkl' #should not be the test model
        deit_trained_dgbed=load_model(load_model_path)
        self.network=MCVisionTransformer(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_cls_emb=num_domains)
        self.network.load_state_dict(deit_trained_dgbed.network.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

       
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss=0
        predictions=torch.chunk(self.predictTrain(all_x),self.num_domains,dim=0)

        for dom in range(self.num_domains):
            loss+= F.cross_entropy(predictions[dom][:,dom], minibatches[dom][1])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predict(self, x):
        return self.network(x)[:,2]
        # return torch.sum(self.network(x),1)
    def predictTrain(self, x):
        return self.network(x)


class DeitSmall_StrongTeachers(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitSmall_StrongTeachers, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        
        self.network=deit_small_distilled_patch16_224(pretrained=True) 
        self.network.head = nn.Linear(384, num_classes)
        self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        teachers=['domainbed/pretrained/single_train_models/ERM/c52ccf0a2ff8fcb27e5f0c637efb5f90/best_val_model_valdom_0_0.9511.pkl','domainbed/pretrained/single_train_models/ERM/c62da0be82f89501d09e0a0e8fd65200/best_val_model_valdom_1_0.9744.pkl','domainbed/pretrained/single_train_models/ERM/cde183b2a672bd049e697054ef75e988/best_val_model_valdom_2_0.9940.pkl','domainbed/pretrained/single_train_models/ERM/6d29d7ebbbc85f0d744e924bb602d5ad/best_val_model_valdom_3_0.9720.pkl']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del teachers[test_envs[-1]]  # Deleting test environment models
        assert len(teachers)==num_domains ,"Required number of teacher pretrained models not given"
        self.Teachnetwork=[load_model(fname).to("cuda") for fname in teachers]
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
    def update(self, minibatches, unlabeled=None):
        train_queues=UpdateClsTrainQueues(minibatches) # Load data class wise
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
 
        cross_learning_data=[]  
        cross_learning_labels=[]
        for dom_n in range(ndomains):
            for i in range(queue_sz) :
                for cls in range(nclass):
                    cross_learning_data.append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=torch.stack(cross_learning_data)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")

        pred,dist=self.predict(cross_learning_data)

        dist=torch.chunk(dist,ndomains,dim=0) 
        loss = F.cross_entropy(pred, cross_learning_labels)
        Wd=self.Wd
        t=self.temp
        cross_learning_data=torch.chunk(cross_learning_data,ndomains,dim=0)
        for dom in range(ndomains):
            divLs= Wd* F.kl_div(
                F.log_softmax(dist[dom] / t, dim=1),
                F.log_softmax(self.predictTeacher(self.Teachnetwork[dom],cross_learning_data[dom]) / t, dim=1),
                reduction='sum',
                log_target=True
            ) * (t * t) / dist[dom].numel()
            loss +=  divLs
      


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)

class DeitSmall_StrongTeachers_nodist(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small) with seperate strong teachers
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitSmall_StrongTeachers_nodist, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        
        self.network=deit_small_patch16_224(pretrained=True) 
        self.network.head = nn.Linear(384, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        teachers=['domainbed/pretrained/single_train_models/ERM/c52ccf0a2ff8fcb27e5f0c637efb5f90/best_val_model_valdom_0_0.9511.pkl','domainbed/pretrained/single_train_models/ERM/c62da0be82f89501d09e0a0e8fd65200/best_val_model_valdom_1_0.9744.pkl','domainbed/pretrained/single_train_models/ERM/cde183b2a672bd049e697054ef75e988/best_val_model_valdom_2_0.9940.pkl','domainbed/pretrained/single_train_models/ERM/6d29d7ebbbc85f0d744e924bb602d5ad/best_val_model_valdom_3_0.9720.pkl']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del teachers[test_envs[-1]]  # Deleting test environment models
        assert len(teachers)==num_domains ,"Required number of teacher pretrained models not given"
        self.Teachnetwork=[load_model(fname).to("cuda") for fname in teachers]
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
    def update(self, minibatches, unlabeled=None):
        train_queues = queue_var.train_queues
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        for id_c in range(nclass): # loop over classes
            for id_d in range(ndomains): # loop over domains
                mb_ids=(minibatches[id_d][1] == id_c).nonzero(as_tuple=True)[0]
                # indices of those egs from domain id_d, whose class label is id_c
                if mb_ids.size(0)==0:
                    continue
                data_tensor=minibatches[id_d][0][mb_ids] # data
                data_tensor = data_tensor.detach()
                # update queue for this class and this domain
                current_queue = train_queues[id_c][id_d]
                current_queue = torch.cat((current_queue, data_tensor), 0)
                current_queue = current_queue[-queue_sz:] # keep only the last queue_sz entries
                train_queues[id_c][id_d] = current_queue
 
        cross_learning_data=[]  
        cross_learning_labels=[]
        for dom_n in range(ndomains):
            for i in range(queue_sz) :
                for cls in range(nclass):
                    cross_learning_data.append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=torch.stack(cross_learning_data)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")

        pred=self.predict(cross_learning_data)

        dist=torch.chunk(pred,ndomains,dim=0) 
        loss = F.cross_entropy(pred, cross_learning_labels)
        Wd=self.Wd
        t=self.temp
        cross_learning_data=torch.chunk(cross_learning_data,ndomains,dim=0)
        for dom in range(ndomains):
            divLs= Wd* F.kl_div(
                F.log_softmax(dist[dom] / t, dim=1),
                F.log_softmax(self.predictTeacher(self.Teachnetwork[dom],cross_learning_data[dom]) / t, dim=1),
                reduction='sum',
                log_target=True
            ) * (t * t) / dist[dom].numel()
            loss +=  divLs
      


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)

class Deit_dist(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_dist,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # self.network=return_backbone_network("DinoSmall",num_classes,self.hparams,add_di_token=False,distilled=True,num_dist_token=10)
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=10,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        # teachers=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        # print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,pred_dist=self.predictTrain(all_x)
        pred_dist=torch.mean(pred_dist,1)
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd


        # changed for self distillation
        loss+= Wd* F.kl_div(
            F.log_softmax(pred_dist / t, dim=1),
            F.log_softmax(pred/t, dim=1),
            reduction='sum',
            log_target=True
        ) * (t * t) / pred_dist.numel()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out,out_dist= self.network(x)
        return out

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
        # text_inputs_text = [(f"a photo of a {PACS_classes[c]}") for c in all_y]
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {PACS_classes[c]}") for c in all_y]).to("cuda")
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

class Vit_untrained_teacher_distill_attn(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Vit_untrained_teacher_distill_attn,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.Teachnetwork=return_backbone_network("DinoSmall",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=10)
        self.network = deit_small_patch16_224(pretrained=True)
        self.network.head = nn.Linear(384, num_classes)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        
        

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,all_attn=self.predictTrain(all_x)
        last_attn=all_attn[-1]
        last_attn_head0=last_attn[:,0,0,1:]
        # print(last_attn_head0.shape)
        # last_attn_=last_attn[:,:,:]
        Teacher_attention=self.predictTeacher(self.Teachnetwork,all_x)
        Teacher_attention = Teacher_attention[:, :, 0, 1:]
        Teacher_attention=torch.mean(Teacher_attention,dim=1)
        loss+= F.cross_entropy(pred,all_y)
        # t = self.temp
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

        # attn_loss=Wd*(F.kl_div(torch.clamp(F.softmax(last_attn_head0, dim=1), 1e-7, 1).log(),F.log_softmax(Teacher_attention, dim=1), reduction='batchmean'))
        

        attn_loss= Wd* F.kl_div(
            F.log_softmax(last_attn_head0 , dim=1),
            F.log_softmax(Teacher_attention, dim=1),
            reduction='batchmean',
            log_target=True
        ) 

        # print(attn_loss)
        loss+=attn_loss
        # loss+= Wd* F.kl_div(
        #     F.log_softmax(dtok_feat / t, dim=1),
        #     F.log_softmax(predTeacher/t, dim=1),
        #     reduction='sum',
        #     log_target=True
        # ) * (t * t) / predTeacher.numel()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x,return_attention=True)
    def predictTeacher(self,net, x):
        return net.get_last_selfattention(x)
        
    def predict(self, x):
        out= self.network(x)
        return out

class Deit_dist_block(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_dist_block,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # self.network=return_backbone_network("DinoSmall",num_classes,self.hparams,add_di_token=False,distilled=True,num_dist_token=10)
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        # teachers=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        # print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred_all,pred_dist_all=self.predictTrain(all_x)
        pred=pred_all[-1]

        random_num=torch.randint(0, len(pred_all)-1, (1,)).item()
        pred_rand=pred_all[random_num]
        # print(len(pred_dist_all))
        pred_dist=pred_dist_all[-1]
        # print(pred_dist.shape)
        pred_dist=torch.mean(pred_dist,1)
        
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd


        # changed for self distillation
        loss+= Wd* F.kl_div(
            F.log_softmax(pred_rand / t, dim=1),
            F.log_softmax(pred/t, dim=1),
            reduction='sum',
            log_target=True
        ) * (t * t) / pred_rand.numel()

        loss+= Wd* F.kl_div(
            F.log_softmax(pred_dist / t, dim=1),
            F.log_softmax(pred/t, dim=1),
            reduction='sum',
            log_target=True
        ) * (t * t) / pred_rand.numel()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x,ret_all_blocks=True)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out,out_dist= self.network(x)
        return out

class Vit_dist_self_teacher(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Vit_dist_self_teacher,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # self.network=return_backbone_network("DinoSmall",num_classes,self.hparams,add_di_token=False,distilled=True,num_dist_token=10)
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=10,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        teachers_PACS=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        teachers_VLCS=['domainbed/pretrained/VLCS/ERM/f9a56b59075f90e98ba6a746f020b111/best_val_model_testdom_[0]_0.8095.pkl','domainbed/pretrained/VLCS/ERM/59f2cefdd9db539bfa26ec5116ddaaa9/best_val_model_testdom_[1]_0.8914.pkl','domainbed/pretrained/VLCS/ERM/74752172f616ce9b3066d289c8461f54/best_val_model_testdom_[2]_0.8754.pkl','domainbed/pretrained/VLCS/ERM/241e42e44bf8761f8b90b71f1c0b13c4/best_val_model_testdom_[3]_0.8567.pkl']
        #below is in cluster
        teachers_OfficeHome=['domainbed/new_outputs/ERM/OfficeHome/b33477e9be6bef3c16b71d28471e95b2/best_val_model_testdom_[0]_0.8374.pkl','domainbed/new_outputs/ERM/OfficeHome/e8bd617010f3fce1a97164ab517f8d61/best_val_model_testdom_[1]_0.8248.pkl','domainbed/new_outputs/ERM/OfficeHome/70131225944c96a0697686f8441feef3/best_val_model_testdom_[2]_0.7817.pkl','domainbed/new_outputs/ERM/OfficeHome/75feeea74a86e5c489f80206f390bc9f/best_val_model_testdom_[3]_0.8129.pkl']
        teachers_TerraIncognita=['domainbed/new_outputs/ERM/TerraIncognita/cd8c6937729e71972c5c830118da6e23/best_val_model_testdom_[0]_0.8952.pkl','domainbed/new_outputs/ERM/TerraIncognita/e9ceb2b0004f4b3762859963b0572cb6/best_val_model_testdom_[1]_0.9032.pkl','domainbed/new_outputs/ERM/TerraIncognita/1e0b2b72dc71e679249646ea44bef5f0/best_val_model_testdom_[2]_0.9084.pkl','domainbed/new_outputs/ERM/TerraIncognita/e239da6cb66555d359e575fac5f4951b/best_val_model_testdom_[3]_0.9221.pkl']
        teachers_DomainNet=['domainbed/new_outputs/ERM/DomainNet/1a0e9071a204bff4c54128bb07ba07a0/best_val_model_testdom_[0]_0.5593.pkl','domainbed/new_outputs/ERM/DomainNet/9211d645d4c895631c553ce52aa87de9/best_val_model_testdom_[1]_0.6320.pkl','domainbed/new_outputs/ERM/DomainNet/d2ea915f855e3d5b5f1b5deef508c00a/best_val_model_testdom_[2]_0.5665.pkl','domainbed/new_outputs/ERM/DomainNet/4f3c2ed530541d05ca34252b0750245a/best_val_model_testdom_[3]_0.5879.pkl','domainbed/new_outputs/ERM/DomainNet/4d47926c474c0640ae26044435d67a2a/best_val_model_testdom_[4]_0.5510.pkl','domainbed/new_outputs/ERM/DomainNet/c0d4b937152a7017aa616a399f4e21e5/best_val_model_testdom_[5]_0.5703.pkl']
        teachers=teachers_PACS
        
        print(teachers[queue_var.current_test_env[0]])
        self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
        

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,pred_dist=self.predictTrain(all_x)
        pred_dist=torch.mean(pred_dist,1)
        predTeacher=self.predictTeacher(self.Teachnetwork,all_x)
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd


        # changed for self distillation
        loss+= Wd* F.kl_div(
            F.log_softmax(pred_dist / t, dim=1),
            F.log_softmax(pred/t, dim=1),
            reduction='sum',
            log_target=True
        ) * (t * t) / pred_dist.numel()

        loss+= Wd* F.kl_div(
            F.log_softmax(pred_dist / t, dim=1),
            F.log_softmax(predTeacher/t, dim=1),
            reduction='sum',
            log_target=True
        ) * (t * t) / pred_dist.numel()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out,out_dist= self.network(x)
        return out

class Vit_dist_zipf(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Vit_dist_zipf,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # self.network=return_backbone_network("DinoSmall",num_classes,self.hparams,add_di_token=False,distilled=True,num_dist_token=10)
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=0,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred=self.predictTrain(all_x)
        # pred_dist=torch.mean(pred_dist,1)
        # predTeacher=self.predictTeacher(self.Teachnetwork,all_x)
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd


        # changed for self distillation
        loss+= Wd*zipf_loss(pred,pred,  all_y,feats=None, loss_mask=False, dense=False)

        # loss+= Wd* F.kl_div(
        #     F.log_softmax(pred_dist / t, dim=1),
        #     F.log_softmax(predTeacher/t, dim=1),
        #     reduction='sum',
        #     log_target=True
        # ) * (t * t) / pred_dist.numel()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out= self.network(x)
        return out

class Vit_dist_zipf_dense(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Vit_dist_zipf_dense,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # self.network=return_backbone_network("DinoSmall",num_classes,self.hparams,add_di_token=False,distilled=True,num_dist_token=10)
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=0,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,pred_patches=self.predictTrain(all_x)
        # print(pred_patches.shape)
        # pred_dist=torch.mean(pred_dist,1)
        # predTeacher=self.predictTeacher(self.Teachnetwork,all_x)
        loss+= F.cross_entropy(pred,all_y)
        # t = self.temp
        Wd=self.Wd


        # changed for self distillation
        loss+= Wd*zipf_loss(pred,pred,  all_y,feats=torch.argmax(pred_patches,dim=2).to("cuda") , loss_mask=False, dense=True)

        # loss+= Wd* F.kl_div(
        #     F.log_softmax(pred_dist / t, dim=1),
        #     F.log_softmax(predTeacher/t, dim=1),
        #     reduction='sum',
        #     log_target=True
        # ) * (t * t) / pred_dist.numel()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x,all_token_logits=True)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out= self.network(x)
        return out

class Vit_with_part_learning(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Vit_with_part_learning,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # self.network=return_backbone_network("DinoSmall",num_classes,self.hparams,add_di_token=False,distilled=True,num_dist_token=10)
        num_classes=num_classes+1
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
        
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=0,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']

        self.norm_trans = transforms.Compose([
            nn.AvgPool2d(51, stride=10),
            transforms.Resize((224,224)),
            # transforms.GaussianBlur(kernel_size=(51,51), sigma=(3.0,3.0))
        ])
        
        self.num_classes=num_classes
        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # print(all_x.shape)
        background=torch.cat([x[:5,:,:,:] for x,y in minibatches])
        # background=torch.randn()
        # print(background.shape)
        background=self.norm_trans(background)
        bacground_labels=torch.full((5*self.num_domains,),self.num_classes-1).to("cuda")
        all_x=torch.cat([all_x,background])
        all_y=torch.cat([all_y,bacground_labels])
        length=3
        patch_size=16
        tot_num_patches=14
        top_left=torch.randint(tot_num_patches-length+1,(2,))
        # left=torch.tensor([13,13])
        transx=torchvision.transforms.functional.crop(all_x, top_left[0].item()*patch_size, top_left[1].item()*patch_size, length*patch_size, length*patch_size) 
        trans=torchvision.transforms.Resize((224,224))
        transx=trans(transx)
        
        if(self.cnt==0):
            batch_images = make_grid(all_x, nrow=7, normalize=True)
            save_image(batch_images, "./domainbed/batchimg_ori.png",normalize=False)
            batch_images = make_grid(transx, nrow=7, normalize=True)
            save_image(batch_images, "./domainbed/batchimg.png",normalize=False)
        loss=0
        pred,pred_patches=self.predictTrain(all_x)
        pred_crop,_=self.predictTrain(transx)
        pred_patches=einops.rearrange(pred_patches,'B (N1 N) C -> B C N1 N',N=tot_num_patches,N1=tot_num_patches)
        pred_sel=torchvision.transforms.functional.crop(pred_patches, top_left[0].item(), top_left[1].item(), length, length) 
        pred_sel=einops.reduce(pred_sel,'B C N1 N -> B C','mean')

        # print(pred_patches.shape)
        # pred_dist=torch.mean(pred_dist,1)
        # predTeacher=self.predictTeacher(self.Teachnetwork,all_x)
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd


        # changed for self distillation
        # loss+= Wd*zipf_loss(pred,pred,  all_y,feats=torch.argmax(pred_patches,dim=2).to("cuda") , loss_mask=False, dense=True)

        kl_loss= Wd* F.kl_div(
            F.log_softmax(pred_sel / t, dim=1),
            F.log_softmax(pred_crop/t, dim=1),
            reduction='sum',
            log_target=True
        ) * (t * t) / pred_crop.numel()
        print(kl_loss)
        loss+=kl_loss
        self.cnt+=1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x,all_token_logits=True)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out= self.network(x)
        return out

class Deit_domain_token(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_domain_token,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        # self.network=return_backbone_network("DinoSmall",num_classes,self.hparams,add_di_token=False,distilled=True,num_dist_token=10)
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask,num_domains_to_tok=3)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 

        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(  
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        # teachers=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        # print(teachers[queue_var.current_test_env[0]])
        # self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])

        all_domains=torch.cat([torch.full(y.shape,i) for i,(x,y) in enumerate(minibatches)]).to("cuda")
        # print(all_domains)

        loss=0
        pred,pred_dist=self.predictTrain(all_x)
        pred_dist=torch.mean(pred_dist,1)
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd

        loss+= F.cross_entropy(pred_dist,all_domains)
        # changed for self distillation
        # loss+= Wd* F.kl_div(
        #     F.log_softmax(pred_dist / t, dim=1),
        #     F.log_softmax(pred/t, dim=1),
        #     reduction='sum',
        #     log_target=True
        # ) * (t * t) / pred_dist.numel()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out,out_dist= self.network(x)
        return out

class CNN_transformer_DI(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CNN_transformer_DI,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        attn_sep_mask=self.hparams["attn_sep_mask"]

        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # self.featurizer.network.avgpool=nn.Identity()
        self.featurizer =nn.Sequential(*list(torchvision.models.resnet50(pretrained=True).children())[:-2])
        
        
        self.vit=dit(img_size=7,num_classes=num_classes, distilled=False,patch_size=1, embed_dim=2048, depth=1, num_heads=8,in_chans=1,add_di_token=True,attn_sep_mask=attn_sep_mask,no_embeddings=True)
        self.network = nn.Sequential(self.featurizer, self.vit)
  

        # printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
       
        self.Wd=self.hparams['Wd']
        
    

    def update(self, minibatches, unlabeled=None):
        

        train_queues=UpdateClsTrainQueues(minibatches) # Load data class wise
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        
        cross_learning_data=[]  
        cross_learning_labels=[]
        for dom_n in range(ndomains):
            for i in range(queue_sz) :
                for cls in range(nclass):
                    cross_learning_data.append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=torch.stack(cross_learning_data)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")

        loss=0
        pred,dtok_feat=self.predictTrain(cross_learning_data)

        loss+= F.cross_entropy(pred,cross_learning_labels)

        Wd=self.Wd
    
        dtok_feat=torch.chunk(dtok_feat,chunks=self.num_domains,dim=0)
   
        domft0,domft1,domft2 = F.softmax(dtok_feat[0], dim=1), F.softmax(dtok_feat[1], dim=1), F.softmax(dtok_feat[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1).log()
        loss += Wd * (F.kl_div(domInvFtAvg, domft0, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft1, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft2, reduction='batchmean')) / 3.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out,out_dist= self.network(x)
        return out

class Deit_DI_tokening(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_DI_tokening,self).__init__(input_shape, num_classes, num_domains,
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
        # self.class_counts=torch.zeros(num_classes)
        # self.class_features.requires_grad=False

    def update(self, minibatches, unlabeled=None):
        class_features=Variable(torch.Tensor(self.class_features)).to('cuda')
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])


        loss=0
        pred,dtok_feat=self.predictTrain(all_x)
        loss+= F.cross_entropy(pred,all_y)
    
        Wd=self.Wd

        # print(dtok_feat)
        # for i in range(len(all_y)):
        #     self.class_features[all_y[i]]+=dtok_feat[i]
        #     self.class_counts[all_y[i]]+=1
        # torch.index_select(self.class_features,0,all_y)+=dtok_feat
        dtok_feat=F.softmax(dtok_feat, dim=1)
        class_features=class_features.index_add_(0,all_y,dtok_feat)
        class_counts=torch.bincount(all_y,minlength=self.num_classes)
        if(self.cnt>=0):
            class_counts=torch.add(class_counts,1)
        else:
            class_counts[class_counts==0] = 1

        # print(class_counts)
        
        class_features=torch.div(class_features,class_counts.reshape(-1,1))
   
        domInvFeatures=torch.index_select(class_features,0,all_y)
        # print(dtok_feat)
        # dtok_feat=torch.chunk(dtok_feat[:,0],chunks=self.num_domains,dim=0)
        # domft0,domft1,domft2 = F.softmax(dtok_feat[0], dim=1), F.softmax(dtok_feat[1], dim=1), F.softmax(dtok_feat[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        loss+=Wd*(F.kl_div(torch.clamp(domInvFeatures, 1e-7, 1).log(),dtok_feat, reduction='batchmean'))
        # print(klloss)
        # print(loss)
        # domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1).log()
        # loss += Wd * (F.kl_div(domInvFtAvg, domft0, reduction='batchmean') +
        #                 F.kl_div(domInvFtAvg, domft1, reduction='batchmean') +
        #                 F.kl_div(domInvFtAvg, domft2, reduction='batchmean')) / 3.
        
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

class Deit_augmix_seperate(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_augmix_seperate,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
    
        attn_sep_mask=self.hparams["attn_sep_mask"]
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        # self.network=dit(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,in_chans=3,add_di_token=True,attn_sep_mask=attn_sep_mask)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Wd=self.hparams['Wd']


    def update(self, minibatches, unlabeled=None):

        train_queues=UpdateClsTrainQueues(minibatches) # Load data class wise
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        
        cross_learning_data=[]  
        cross_learning_labels=[]
        for dom_n in range(ndomains):
            for i in range(queue_sz) :
                for cls in range(nclass):
                    cross_learning_data.append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=torch.stack(cross_learning_data)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")

        loss=0
        pred,_,_,dtok_feat=self.predictTrain(cross_learning_data)
        loss+= F.cross_entropy(pred,cross_learning_labels)
    
        Wd=self.Wd

        dtok_feat=torch.chunk(dtok_feat[:,0],chunks=self.num_domains,dim=0)
        domft0,domft1,domft2 = F.softmax(dtok_feat[0], dim=1), F.softmax(dtok_feat[1], dim=1), F.softmax(dtok_feat[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1).log()
        loss += Wd * (F.kl_div(domInvFtAvg, domft0, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft1, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft2, reduction='batchmean')) / 3.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x,return_cls_dist_feat=True)

    def predict(self, x):
        out,out_dist= self.network(x,return_dist_inf=True)
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

class DI_tokening_aver(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DI_tokening_aver,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        self.split_class=7
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
        self.num_class_select=self.hparams['num_class_select']
        self.cnt=0


    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # all_y=einops.rearrange(all_y,'(n c s)->(c s n)',n=7 ,c=3, s=7)
        # all_x=einops.rearrange(all_x,'(n c s) C H W->(c s n) C H W ',n=7 ,c=3, s=7)

        
        # batch_images = make_grid(all_x, nrow=7, normalize=True)
        # if(self.cnt==0):
        #     save_image(batch_images, "./domainbed/batchimg.png",normalize=False)
        loss=0
        pred,dtok_feat=self.predictTrain(all_x)
        loss+= F.cross_entropy(pred,all_y)
    
        Wd=self.Wd

        dtok_feat=torch.chunk(dtok_feat,chunks=self.split_class,dim=0)
        for cls in range(len(dtok_feat)):
 
            feat=F.softmax(dtok_feat[cls], dim=1)
            avg=torch.mean(feat,dim=0).expand(feat.shape[0],-1)
            loss += (Wd/(1.0*self.split_class)) * (F.kl_div(feat.log(),torch.clamp(avg, 1e-7, 1) , reduction='batchmean'))
                        

        # domft0,domft1,domft2 = F.softmax(dtok_feat[0], dim=1), F.softmax(dtok_feat[1], dim=1), F.softmax(dtok_feat[2], dim=1)
        
        # # Clamp mixture distribution to avoid exploding KL divergence
        # domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1)
        # loss += Wd * (F.kl_div(domft0.log(),domInvFtAvg , reduction='batchmean') +
        #                 F.kl_div(domft1.log(),domInvFtAvg,  reduction='batchmean') +
        #                 F.kl_div(domft2.log(),domInvFtAvg,  reduction='batchmean')) / 3.

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

class DeitBase_augmix_seperate(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitBase_augmix_seperate,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        network_deit=deit_base_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(768, num_classes)
    
        attn_sep_mask=self.hparams["attn_sep_mask"]
        
        # self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
        self.network=dit(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=768, depth=12, num_heads=12,in_chans=3,add_di_token=True,attn_sep_mask=attn_sep_mask)
        
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Wd=self.hparams['Wd']


    def update(self, minibatches, unlabeled=None):

        train_queues=UpdateClsTrainQueues(minibatches) # Load data class wise
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        
        cross_learning_data=[]  
        cross_learning_labels=[]
        for dom_n in range(ndomains):
            for i in range(queue_sz) :
                for cls in range(nclass):
                    cross_learning_data.append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=torch.stack(cross_learning_data)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")

        loss=0
        pred,dtok_feat=self.predictTrain(cross_learning_data)
        loss+= F.cross_entropy(pred,cross_learning_labels)
    
        Wd=self.Wd

        dtok_feat=torch.chunk(dtok_feat,chunks=self.num_domains,dim=0)
        domft0,domft1,domft2 = F.softmax(dtok_feat[0], dim=1), F.softmax(dtok_feat[1], dim=1), F.softmax(dtok_feat[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1).log()
        loss += Wd * (F.kl_div(domInvFtAvg, domft0, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft1, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft2, reduction='batchmean')) / 3.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)

    def predict(self, x):
        out,out_dist= self.network(x)
        return out

class Deit_DomInv(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_DomInv,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384*2, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
        mask_clsT_distT=self.hparams["mask_clsT_distT"]
        self.network=DGVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=2,attn_sep_mask=attn_sep_mask,mask_clsT_distT=mask_clsT_distT)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']

        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/outputs/PACS/erm/56ca19b798087be4998b8b46ef3281f7/best_val_model_testdom_[0]_0.9709.pkl','domainbed/outputs/PACS/erm/10b4762ab54115af03884b99e5a136ed/best_val_model_testdom_[1]_0.9690.pkl','domainbed/outputs/PACS/erm/89d41edbd75abb12dcf9fd0bfc1061ed/best_val_model_testdom_[2]_0.9591.pkl','domainbed/outputs/PACS/erm/0d52a60924ea5be50a0ca63e1ad8d55e/best_val_model_testdom_[3]_0.9674.pkl']
        print(teachers[queue_var.current_test_env[0]])
        self.Teachnetwork=load_model(teachers[queue_var.current_test_env[0]]).to("cuda") 
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        train_queues=UpdateClsTrainQueues(minibatches) # Load data class wise
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        
        cross_learning_data=[]  
        cross_learning_labels=[]
        for dom_n in range(ndomains):
            for i in range(queue_sz) :
                for cls in range(nclass):
                    cross_learning_data.append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=torch.stack(cross_learning_data)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")


        loss=0
        pred,pred_dist,dom_inv_feat,dom_spec_feat=self.predict(cross_learning_data)
        pred_dist=pred_dist
        loss+= F.cross_entropy(pred,cross_learning_labels)
        t = self.temp
        Wd=self.Wd

        loss+= Wd* F.kl_div(
            F.log_softmax(pred_dist / t, dim=1),
            F.log_softmax(pred / t, dim=1),
            reduction='sum',
            log_target=True
        ) * (t * t) / pred_dist.numel()

        dom_inv_feat_ch=torch.chunk(dom_inv_feat,chunks=ndomains,dim=0)
        domft0,domft1,domft2 = F.softmax(dom_inv_feat_ch[0], dim=1), F.softmax(dom_inv_feat_ch[1], dim=1), F.softmax(dom_inv_feat_ch[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1).log()
        loss += 25 * (F.kl_div(domInvFtAvg, domft0, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft1, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft2, reduction='batchmean')) / 3.
        
        # mdi_z = torch.mean(dom_inv_feat, 0)
        # mds_z = torch.mean(dom_spec_feat, 0)

        # di_z_n = dom_inv_feat - mdi_z[None, :]
        # ds_z_n = dom_spec_feat - mds_z[None, :]
        # C = di_z_n[:, :, None] * ds_z_n[:, None, :]

        # target_cr = torch.zeros(C.shape[0], C.shape[1], C.shape[2]).cuda()
        # disentangle_loss = nn.MSELoss()(C, target_cr)

        # loss+=25.0*disentangle_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predict(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    # def predict(self, x):
    #     out,out_dist= self.network(x,return_dist_inf=True)
    #     return out

class Deit_simple_augmix(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_simple_augmix,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.network = deit_small_patch16_224(pretrained=True)
        self.network.head = nn.Linear(384, num_classes)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.num_domains=num_domains

        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']


    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        train_queues=UpdateClsTrainQueues(minibatches) # Load data class wise
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        
        cross_learning_data=[]  
        cross_learning_labels=[]
        for dom_n in range(ndomains):
            for i in range(queue_sz) :
                for cls in range(nclass):
                    cross_learning_data.append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=torch.stack(cross_learning_data)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")


        loss=0
        pred,features=self.predict(cross_learning_data)
        loss+= F.cross_entropy(pred,cross_learning_labels)
        t = self.temp
        Wd=self.Wd

        
        features=torch.chunk(features,chunks=ndomains,dim=0)
        domft0,domft1,domft2 = F.softmax(features[0], dim=1), F.softmax(features[1], dim=1), F.softmax(features[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1).log()
        loss += Wd * (F.kl_div(domInvFtAvg, domft0, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft1, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft2, reduction='batchmean')) / 3.
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}



    def predict(self, x):
        return self.network(x,return_cls_feat=True)


class MultiDomainDistillation_Dtokens(Algorithm):
    """
    MultiDomainDistillation_Dtokens; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MultiDomainDistillation_Dtokens, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
        mask_clsT_distT=self.hparams["mask_clsT_distT"]
        mask_dist_other_patches=self.hparams["mask_dist_other_patches"]
        # mask_clsT_distT=True
        # mask_dist_other_patches=False
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=num_domains,attn_sep_mask=attn_sep_mask,mask_clsT_distT=mask_clsT_distT,mask_dist_other_patches=mask_dist_other_patches)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/pretrained/single_train_models/DeitSmall/9788f5f1331a757f0112c9023693ca86/best_val_model_valdom_0_0.9633.pkl','domainbed/pretrained/single_train_models/DeitSmall/a3e438e97c3857530837fbccedc08cf8/best_val_model_valdom_1_0.9786.pkl','domainbed/pretrained/single_train_models/DeitSmall/820147d8f3bc2473e6b839f2e4fb0f2e/best_val_model_valdom_2_0.9940.pkl','domainbed/pretrained/single_train_models/DeitSmall/5fb01bd30f2f20383bd9b6d6ad32a716/best_val_model_valdom_3_0.9694.pkl']
        # teachers=['domainbed/pretrained/single_train_models/ERM/c52ccf0a2ff8fcb27e5f0c637efb5f90/best_val_model_valdom_0_0.9511.pkl','domainbed/pretrained/single_train_models/ERM/c62da0be82f89501d09e0a0e8fd65200/best_val_model_valdom_1_0.9744.pkl','domainbed/pretrained/single_train_models/ERM/cde183b2a672bd049e697054ef75e988/best_val_model_valdom_2_0.9940.pkl','domainbed/pretrained/single_train_models/ERM/6d29d7ebbbc85f0d744e924bb602d5ad/best_val_model_valdom_3_0.9720.pkl']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del teachers[test_envs[-1]]  # Deleting test environment models
        assert len(teachers)==num_domains ,"Required number of teacher pretrained models not given"
        self.Teachnetwork=[load_model(fname).to("cuda") for fname in teachers]
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss=0
        pred,pred_dist=self.predictTrain(all_x)
        loss+= F.cross_entropy(pred,all_y)
        pred_dist=torch.chunk(pred_dist,self.num_domains,dim=0)
        t = self.temp
        Wd=self.Wd
        for dom in range(self.num_domains):
            
            loss+= Wd* F.kl_div(
                F.log_softmax(pred_dist[dom][:,dom] / t, dim=1),
                F.log_softmax(self.predictTeacher(self.Teachnetwork[dom],minibatches[dom][0]) / t, dim=1),
                reduction='sum',
                log_target=True
            ) * (t * t) / pred_dist[dom][:,dom].numel()
    

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out,out_dist= self.network(x,return_dist_inf=True)
        return out
        m=nn.Softmax(dim=1)
        weighted_out=-torch.sum(torch.nn.functional.log_softmax(out,dim=1)*m(out),dim=1).view(-1,1).expand(-1,out.shape[1])*out
        for i in range(out_dist.shape[1]):
            weighted_out+=-torch.sum(torch.nn.functional.log_softmax(out_dist[:,i],dim=1)*m(out_dist[:,i]),dim=1).view(-1,1).expand(-1,out.shape[1])*out_dist[:,i]
        return weighted_out        
        # weighted_out=torch.max(m(out),1).values.view(-1,1).expand(-1,out.shape[1])*out
        # for i in range(out_dist.shape[1]):
        #     weighted_out+=torch.max(m(out_dist[:,i]),1).values.view(-1,1).expand(-1,out.shape[1])*out_dist[:,i]
        # return weighted_out
        out,out_dist= self.network(x,return_dist_inf=True)
        return out

class MultiDomainDistillation_Dtokens_patchmask(Algorithm):
    """
    MultiDomainDistillation_Dtokens; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MultiDomainDistillation_Dtokens_patchmask, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
        mask_clsT_distT=self.hparams["mask_clsT_distT"]
        mask_dist_other_patches=self.hparams["mask_dist_other_patches"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=num_domains,attn_sep_mask=attn_sep_mask,mask_clsT_distT=mask_clsT_distT,mask_dist_other_patches=mask_dist_other_patches)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/pretrained/single_train_models/ERM/c52ccf0a2ff8fcb27e5f0c637efb5f90/best_val_model_valdom_0_0.9511.pkl','domainbed/pretrained/single_train_models/ERM/c62da0be82f89501d09e0a0e8fd65200/best_val_model_valdom_1_0.9744.pkl','domainbed/pretrained/single_train_models/ERM/cde183b2a672bd049e697054ef75e988/best_val_model_valdom_2_0.9940.pkl','domainbed/pretrained/single_train_models/ERM/6d29d7ebbbc85f0d744e924bb602d5ad/best_val_model_valdom_3_0.9720.pkl']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del teachers[test_envs[-1]]  # Deleting test environment models
        assert len(teachers)==num_domains ,"Required number of teacher pretrained models not given"
        self.Teachnetwork=[load_model(fname).to("cuda") for fname in teachers]
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss=0
        pred,pred_dist=self.predictTrain(all_x)
        loss+= F.cross_entropy(pred,all_y)
        pred_dist=torch.chunk(pred_dist,self.num_domains,dim=0)
        t = self.temp
        Wd=self.Wd
        for dom in range(self.num_domains):
            
            loss+= Wd* F.kl_div(
                F.log_softmax(pred_dist[dom][:,dom] / t, dim=1),
                F.log_softmax(self.predictTeacher(self.Teachnetwork[dom],minibatches[dom][0]) / t, dim=1),
                reduction='sum',
                log_target=True
            ) * (t * t) / pred_dist[dom][:,dom].numel()
    

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out,out_dist= self.network(x,return_dist_inf=True)
        return out_dist[:,2]
        m=nn.Softmax(dim=1)
        weighted_out=-torch.sum(torch.nn.functional.log_softmax(out,dim=1)*m(out),dim=1).view(-1,1).expand(-1,out.shape[1])*out
        for i in range(out_dist.shape[1]):
            weighted_out+=-torch.sum(torch.nn.functional.log_softmax(out_dist[:,i],dim=1)*m(out_dist[:,i]),dim=1).view(-1,1).expand(-1,out.shape[1])*out_dist[:,i]
        return weighted_out        
        # weighted_out=torch.max(m(out),1).values.view(-1,1).expand(-1,out.shape[1])*out
        # for i in range(out_dist.shape[1]):
        #     weighted_out+=torch.max(m(out_dist[:,i]),1).values.view(-1,1).expand(-1,out.shape[1])*out_dist[:,i]
        # return weighted_out
        out,out_dist= self.network(x,return_dist_inf=True)
        return out        

class MultiDomainDistillation_Dtokens_CE(Algorithm):
    """
    MultiDomainDistillation_Dtokens; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MultiDomainDistillation_Dtokens_CE, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
        mask_clsT_distT=self.hparams["mask_clsT_distT"]
        mask_dist_other_patches=self.hparams["mask_dist_other_patches"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=num_domains,attn_sep_mask=attn_sep_mask,mask_clsT_distT=mask_clsT_distT,mask_dist_other_patches=mask_dist_other_patches)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/pretrained/single_train_models/ERM/c52ccf0a2ff8fcb27e5f0c637efb5f90/best_val_model_valdom_0_0.9511.pkl','domainbed/pretrained/single_train_models/ERM/c62da0be82f89501d09e0a0e8fd65200/best_val_model_valdom_1_0.9744.pkl','domainbed/pretrained/single_train_models/ERM/cde183b2a672bd049e697054ef75e988/best_val_model_valdom_2_0.9940.pkl','domainbed/pretrained/single_train_models/ERM/6d29d7ebbbc85f0d744e924bb602d5ad/best_val_model_valdom_3_0.9720.pkl']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del teachers[test_envs[-1]]  # Deleting test environment models
        assert len(teachers)==num_domains ,"Required number of teacher pretrained models not given"
        self.Teachnetwork=[load_model(fname).to("cuda") for fname in teachers]
        
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss=0
        pred,pred_dist=self.predictTrain(all_x)
        # loss+= F.cross_entropy(pred,all_y)
        pred_dist=torch.chunk(pred_dist,self.num_domains,dim=0)
        
        t = self.temp
        Wd=self.Wd
        for dom in range(self.num_domains):
            loss+= Wd* F.cross_entropy(pred_dist[dom][:,dom],minibatches[dom][1])


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out,out_dist= self.network(x,return_dist_inf=True)
        return out_dist[:,2]
        # return torch.sum(out_dist,dim=1)
        m=nn.Softmax(dim=1)
        weighted_out=m(out)
        for i in range(out_dist.shape[1]):
            weighted_out+=m(out_dist[:,i])
        return weighted_out

class MDT_self_dist_wCE(Algorithm):
    """MultiDomainDistillation_Dtokens; Order of domains in training should be preserved"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MDT_self_dist_wCE, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)
        
        self.num_domains=num_domains
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
        mask_clsT_distT=self.hparams["mask_clsT_distT"]
        mask_dist_other_patches=self.hparams["mask_dist_other_patches"]

        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=num_domains,attn_sep_mask=attn_sep_mask,mask_clsT_distT=mask_clsT_distT,mask_dist_other_patches=mask_dist_other_patches)
        self.network.load_state_dict(network_deit.state_dict(),strict=False) 
        printNetworkParams(self.network)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']
        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/pretrained/single_train_models/ERM/c52ccf0a2ff8fcb27e5f0c637efb5f90/best_val_model_valdom_0_0.9511.pkl','domainbed/pretrained/single_train_models/ERM/c62da0be82f89501d09e0a0e8fd65200/best_val_model_valdom_1_0.9744.pkl','domainbed/pretrained/single_train_models/ERM/cde183b2a672bd049e697054ef75e988/best_val_model_valdom_2_0.9940.pkl','domainbed/pretrained/single_train_models/ERM/6d29d7ebbbc85f0d744e924bb602d5ad/best_val_model_valdom_3_0.9720.pkl']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del teachers[test_envs[-1]]  # Deleting test environment models
        assert len(teachers)==num_domains ,"Required number of teacher pretrained models not given"
        self.Teachnetwork=[load_model(fname).to("cuda") for fname in teachers]
        self.counte=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss=0
        pred,pred_dist=self.predictTrain(all_x)
        # loss+= F.cross_entropy(pred,all_y)
        pred_dist=torch.chunk(pred_dist,self.num_domains,dim=0)
        pred=torch.chunk(pred,self.num_domains,dim=0)
        t = self.temp
        Wd=self.Wd
        for dom in range(self.num_domains):
            loss+=  F.cross_entropy(pred_dist[dom][:,dom],minibatches[dom][1])
            if(self.counte>300):
                loss+= Wd* F.kl_div(
                    F.log_softmax(pred[dom]/ t, dim=1),
                    F.log_softmax(pred_dist[dom][:,dom] / t, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (t * t) / pred_dist[dom][:,dom].numel()
        self.counte+=1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net.predict(x)
    def predict(self, x):
        out,out_dist= self.network(x,return_dist_inf=True)
        return out
        m=nn.Softmax(dim=1)
        weighted_out=m(out)
        for i in range(out_dist.shape[1]):
            weighted_out+=m(out_dist[:,i])
        return weighted_out       

class MultiDomainDistillation_Dtokens_wtClsDist(MultiDomainDistillation_Dtokens):
    """
    MultiDomainDistillation_Dtokens; Order of domains in training should be preserved
    """


    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss=0
        pred,pred_dist=self.predictTrain(all_x)
        loss+= F.cross_entropy(pred,all_y)
        pred=torch.chunk(pred,self.num_domains,dim=0)
        pred_dist=torch.chunk(pred_dist,self.num_domains,dim=0)
        t = self.temp
        Wd=self.Wd
        for dom in range(self.num_domains):
            
            loss+= Wd* F.kl_div(
                F.log_softmax(pred_dist[dom][:,dom] / t, dim=1),
                F.log_softmax(self.predictTeacher(self.Teachnetwork[dom],minibatches[dom][0]) / t, dim=1),
                reduction='sum',
                log_target=True
            ) * (t * t) / pred_dist[dom][:,dom].numel()

            loss+= Wd* F.kl_div(
                F.log_softmax(pred_dist[dom][:,dom]/t, dim=1),
                F.log_softmax(pred[dom]/t, dim=1),
                reduction='sum',
                log_target=True
            ) * (t * t) / pred_dist[dom][:,dom].numel()
    

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        out,out_dist= self.network(x,return_dist_inf=True)
        m=nn.Softmax(dim=1)
        weighted_out=m(out)*out
        for i in range(out_dist.shape[1]):
            weighted_out+=m(out_dist[:,i])*out_dist[:,i]
        return weighted_out

class Deit_Dino(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_Dino, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
    
        self.network = deit_small_patch16_224(pretrained=True)
        self.network.head = nn.Linear(384, num_classes)
        self.dino_small=return_backbone_network("DinoSmall",num_classes,self.hparams,add_di_token=False,distilled=False,num_dist_token=0)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.cnt=0
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])

        pred,pred_attn=self.predict(all_x)  # For vit small( no dist)
        loss = F.cross_entropy(pred, all_y)
        # loss=0
  
        # print("loss",loss)
        last_attn=pred_attn[-1]
        last_attn=last_attn[:,:,0,1:]
        last_attn_ori=torch.mean(last_attn,dim=1)
        last_attn = last_attn_ori.reshape(-1,1, 14, 14).float()

        attentions_D=self.dino_small.get_last_selfattention(all_x)
        attentions_D = attentions_D[:, :, 0, 1:]
        attentions=torch.mean(attentions_D,dim=1)

        
        # print(last_attn[0])
        threshold=0.75
        val, idx = torch.sort(attentions)
        B=attentions.shape[0]
        val /= torch.sum(val, dim=1, keepdim=True)
        cum_val = torch.cumsum(val, dim=1)
        th_attn = cum_val > (1 - threshold)
        idx2 = torch.argsort(idx)
        for sample in range(B):
            th_attn[sample] = th_attn[sample][idx2[sample]]
     
        th_attn = th_attn.reshape(B,1,14,14).float()

        if(self.cnt%1000==0):
            original_batch = make_grid(all_x, nrow=10, normalize=True)
            save_image(original_batch, "./img_outs/check_attentionsvit/original_grid"+str(self.cnt)+".png", normalize=False)
            last_attn_cpy=last_attn_ori.clone().detach()
            # attentions_img = torch.mean(attentions_D,dim=1).reshape(B,1, 14, 14)
            attentions_img=F.softmax(last_attn_cpy,dim=1).reshape(B,1,14,14)
            attentions_img = nn.functional.interpolate(attentions_img,scale_factor=16, mode="nearest").reshape(B,224,224).to('cpu').numpy()
            # save attentions heatmaps
            nh=1
            for j in range(nh):
                
                # attn_batch = make_grid(torch.tensor(attentions_img), nrow=10, normalize=True)
                # save_image(attn_batch, "./img_outs/check_attentionsvit/attn_grid"+str(self.cnt)+".png", normalize=False)
                for imno in range(B):
                    fname = "./img_outs/check_attentionsvit/"+ "attbatch" + str(self.cnt) +"_"+str(imno)+ ".png"
                    # print(attentions_img[imno].shape)
                    
                    plt.imsave(fname=fname, arr=attentions_img[imno], format='png')
                    # print(f"{fname} saved.")

        # cos=nn.CosineSimilarity(dim=1, eps=1e-6)
        # seg_loss= 1-torch.mean(cos(last_attn_ori, attentions))
        # seg_loss=similarityCE(last_attn_ori,attentions)

        # seg_loss=dice_loss(th_attn.long(),last_attn)
        # seg_loss=F.binary_cross_entropy_with_logits(last_attn, th_attn,pos_weight=torch.tensor(10.0))
        self.cnt+=1
        t=1.0
        Wd=1.0
        seg_loss= Wd* F.kl_div(
                    F.softmax(last_attn_ori/ t, dim=1),
                    F.log_softmax(attentions, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (t * t) / attentions.numel()

        seg_loss+= Wd* F.kl_div(
                    F.log_softmax(attentions, dim=1),
                    F.softmax(last_attn_ori/ t, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (t * t) / attentions.numel()

        loss+=seg_loss
        # print(seg_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
        
    def predict(self, x):
        return self.network(x,return_attention=True)
        return_tokens,return_attention

class Deit_Dino_jac(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_Dino_jac, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
    
        self.network = deit_small_patch16_224(pretrained=True)
        self.network.head = nn.Linear(384, num_classes)
        self.dino_small=load_dino()
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.cnt=0
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])

        pred,pred_attn=self.predict(all_x)  # For vit small( no dist)
        last_attn=pred_attn[-1] # attention from last block
        target_attn=self.dino_small.get_last_selfattention(all_x)

        seg_loss2=get_jaccard_loss_from_attention(last_attn,target_attn)
        # print("jac",seg_loss2 )
    
        loss = F.cross_entropy(pred, all_y)
        loss+=seg_loss2


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
        
    def predict(self, x):
        return self.network(x,return_attention=True)
        return_tokens,return_attention

class Dino(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_Dino_jac, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
    
        self.network = deit_small_patch16_224(pretrained=True)
        self.network.head = nn.Linear(384, num_classes)
        self.dino_small=load_dino()
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.cnt=0
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])

        pred,pred_attn=self.predict(all_x)  # For vit small( no dist)
        last_attn=pred_attn[-1] # attention from last block
        target_attn=self.dino_small.get_last_selfattention(all_x)

        seg_loss2=get_jaccard_loss_from_attention(last_attn,target_attn)
        # print("jac",seg_loss2 )
    
        loss = F.cross_entropy(pred, all_y)
        loss+=seg_loss2


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
        
    def predict(self, x):
        return self.network(x,return_attention=True)
        return_tokens,return_attention

class DeitSmall(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitSmall, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
    
        self.network = deit_small_patch16_224(pretrained=True)
        self.network.head = nn.Linear(384, num_classes)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        pred=self.predict(all_x)
        loss = F.cross_entropy(pred, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
        
    def predict(self, x):
        return self.network(x)

class DeitBase(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitBase, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
    
        self.network = deit_base_patch16_224(pretrained=True)
        self.network.head = nn.Linear(768, num_classes)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        pred=self.predict(all_x)
        loss = F.cross_entropy(pred, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
        
    def predict(self, x):
        return self.network(x)

class DeitTiny(ERM):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitTiny, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
                    
        # self.network = torch.hub.load('/home/computervision1/Sanoojan/DomainBedS/deit',
        #                               'deit_Tiny_patch16_224', pretrained=True, source='local')    
        self.network=deit_tiny_patch16_224(pretrained=False) 
        self.network.head = nn.Linear(192, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
  
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
        )
class T2T(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(T2T, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
                      
        self.network=t2t_vit_t_24(pretrained=True)
        # load_for_transfer_learning(self.network,"domainbed/pretrained/T2T/81.7_T2T_ViTt_14.pth",use_ema=True, strict=True,num_classes=7)
        self.network.head = nn.Linear(512, num_classes)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],

        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        pred=self.predict(all_x)
        loss = F.cross_entropy(pred, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        out= self.network(x)
        # print(out.shape)
        return out

class T2T_augmix_seperate(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(T2T_augmix_seperate, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        attn_sep_mask=self.hparams["attn_sep_mask"]
        self.network=t2t_vit_7(pretrained=True,add_di_tok=True,attn_sep_mask=attn_sep_mask)
        # load_for_transfer_learning(self.network,"domainbed/pretrained/T2T/81.7_T2T_ViTt_14.pth",use_ema=True, strict=True,num_classes=7)
        self.network.head = nn.Linear(256, num_classes) # for t2t-7-> 256 for t2t24->512
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],

        )           
       
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']

    def update(self, minibatches, unlabeled=None):


        train_queues=UpdateClsTrainQueues(minibatches) # Load data class wise
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        
        cross_learning_data=[]  
        cross_learning_labels=[]
        for dom_n in range(ndomains):
            for i in range(queue_sz) :
                for cls in range(nclass):
                    cross_learning_data.append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=torch.stack(cross_learning_data)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")

        loss=0
        pred,dtok_feat=self.predictTrain(cross_learning_data)
        loss+= F.cross_entropy(pred,cross_learning_labels)
        t = self.temp
        Wd=self.Wd

        
        dtok_feat=torch.chunk(dtok_feat,chunks=ndomains,dim=0)
        domft0,domft1,domft2 = F.softmax(dtok_feat[0], dim=1), F.softmax(dtok_feat[1], dim=1), F.softmax(dtok_feat[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1).log()
        loss += Wd * (F.kl_div(domInvFtAvg, domft0, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft1, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft2, reduction='batchmean')) / 3.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        outputs,outputs_di= self.network(x)
        return outputs
    def predictTrain(self, x):
        outputs,outputs_di= self.network(x)
        return outputs,outputs_di

class CVTSmall(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CVTSmall, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
                      
        self.network=small_cvt(pretrained=True) 
        self.network.head = nn.Linear(384, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
  
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],

        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        pred=self.predict(all_x)
        loss = F.cross_entropy(pred, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network(x)[-1]

class CVT_augmix_seperate(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CVT_augmix_seperate, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
                      
        self.network=tiny_cvt(pretrained=True,add_di_tok=True) 
        self.network.head = nn.Linear(384, num_classes)
        attn_sep_mask=self.hparams["attn_sep_mask"]

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],

        )
        self.temp=self.hparams['temp']
        self.Wd=self.hparams['Wd']

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])

        train_queues=UpdateClsTrainQueues(minibatches) # Load data class wise
        nclass=len(train_queues)
        ndomains=len(train_queues[0])
        
        cross_learning_data=[]  
        cross_learning_labels=[]
        for dom_n in range(ndomains):
            for i in range(queue_sz) :
                for cls in range(nclass):
                    cross_learning_data.append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=torch.stack(cross_learning_data)
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")

        loss=0
        pred,dtok_feat=self.predictTrain(cross_learning_data)
        loss+= F.cross_entropy(pred,cross_learning_labels)
        t = self.temp
        Wd=self.Wd

        
        dtok_feat=torch.chunk(dtok_feat,chunks=ndomains,dim=0)
        domft0,domft1,domft2 = F.softmax(dtok_feat[0], dim=1), F.softmax(dtok_feat[1], dim=1), F.softmax(dtok_feat[2], dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        domInvFtAvg = torch.clamp((domft0 + domft1 + domft2) / 3., 1e-7, 1).log()
        loss += Wd * (F.kl_div(domInvFtAvg, domft0, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft1, reduction='batchmean') +
                        F.kl_div(domInvFtAvg, domft2, reduction='batchmean')) / 3.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        outputs,outputs_di= self.network(x)
        return outputs[-1]
    def predictTrain(self, x):
        outputs,outputs_di= self.network(x)
        return outputs[-1],outputs_di[-1]

class CVTTiny(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CVTTiny, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
         
        self.network=tiny_cvt(pretrained=False) 
        self.network.head = nn.Linear(384, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
  
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],

        )
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        pred=self.predict(all_x)
        loss = F.cross_entropy(pred, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network(x)[-1]


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
    
    
