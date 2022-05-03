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
from domainbed.lib.cvt import tiny_cvt,small_cvt
from torch.nn.functional import interpolate
import domainbed.lib.Dino_vit as dino
import itertools
from prettytable import PrettyTable
import copy
import numpy as np
from torchvision.utils import save_image
from torchvision.utils import make_grid
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
    'Deit_Dino_jac'

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
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=1,attn_sep_mask=attn_sep_mask)
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
        loss=0
        pred,pred_dist=self.predictTrain(all_x)
        pred_dist=pred_dist[:,0]
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd

        loss+= Wd* F.kl_div(
            F.log_softmax(pred_dist / t, dim=1),
            F.log_softmax(self.predictTeacher(self.Teachnetwork,all_x) / t, dim=1),
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

class Deit_DomInv(Algorithm):
    """
    Deit_dist; Order of domains in training should be preserved
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Deit_DomInv,self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_domains=num_domains
        network_deit=deit_small_patch16_224(pretrained=True) 
        network_deit.head = nn.Linear(384, num_classes)
        # network_deit.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        attn_sep_mask=self.hparams["attn_sep_mask"]
    
        self.network=MDVisionTransformer(img_size=224,num_classes=num_classes, distilled=True,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_dist_token=2,attn_sep_mask=attn_sep_mask)
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
        
        cross_learning_data=[[] for i in range(ndomains)]  
        cross_learning_labels=[]
        for cls in range(nclass):
            for i in range(queue_sz) :
                for dom_n in range(ndomains):
                    cross_learning_data[dom_n].append(train_queues[cls][dom_n][i])
                    cross_learning_labels.append(cls)

        cross_learning_data=[torch.stack(data) for data in cross_learning_data]
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")



        loss=0
        pred,pred_dist=self.predictTrain(cross_learning_data)
        pred_dist=pred_dist[:,0]
        loss+= F.cross_entropy(pred,all_y)
        t = self.temp
        Wd=self.Wd

        loss+= Wd* F.kl_div(
            F.log_softmax(pred_dist / t, dim=1),
            F.log_softmax(self.predictTeacher(self.Teachnetwork,all_x) / t, dim=1),
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
        out,out_dist= self.network(x,return_dist_inf=True)
        return out

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
        seg_loss=similarityCE(last_attn_ori,attentions)

        # seg_loss=dice_loss(th_attn.long(),last_attn)
        # seg_loss=F.binary_cross_entropy_with_logits(last_attn, th_attn,pos_weight=torch.tensor(10.0))
        self.cnt+=1
        t=1.0
        Wd=1.0
        # seg_loss= Wd* F.kl_div(
        #             F.log_softmax(last_attn_ori/ t, dim=1),
        #             torch.log(attentions),
        #             reduction='sum',
        #             log_target=True
        #         ) * (t * t) / attentions.numel()

        # seg_loss+= Wd* F.kl_div(
        #             torch.log(attentions),
        #             F.LogSoftmax(last_attn_ori/ t, dim=1),
        #             reduction='sum',
        #             log_target=True
        #         ) * (t * t) / attentions.numel()

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
    
        last_attn=pred_attn[-1]
        last_attn=last_attn[:,:,0,1:]
        last_attn = torch.mean(last_attn,dim=1).reshape(-1,1,14, 14).float()
        # last_attn = last_attn.reshape(-1,6,14, 14).float()
        loss = F.cross_entropy(pred, all_y)

        attentions=self.dino_small.get_last_selfattention(all_x)
        attentions = attentions[:, :, 0, 1:]
        attentions=torch.mean(attentions,dim=1)
        threshold=0.75
        val, idx = torch.sort(attentions)
        B=attentions.shape[0]
        val /= torch.sum(val, dim=1, keepdim=True)
        cum_val = torch.cumsum(val, dim=1)
        th_attn = cum_val > (1 - threshold)
        idx2 = torch.argsort(idx)
        for sample in range(B):
            th_attn[sample] = th_attn[sample][idx2[sample]]
        th_attn = th_attn.reshape(B,14,14).long()
        
        
        seg_loss=jaccard_loss(th_attn,last_attn)

        # last_attn=torch.mean(last_attn,dim=1).reshape(-1,196)
        # seg_loss= 1-compute_jaccard(last_attn,attentions)

      
        loss+=seg_loss


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

def load_dino():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = dino.__dict__['vit_small'](patch_size=16, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    pretrained_weights="domainbed/pretrained/dino/dino_deitsmall16_pretrain.pth"

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

def compute_jaccard(attentions_Batch,target_batch):
    # make the image divisible by the patch size
    # attentions_Batch:B,nh,196
    # target_batch:B,nh,196
    w_featmap = 14
    h_featmap = 14
    nh=6
    # th_attn_all=[]
    threshold=0.75
    patch_size=16
    jacc_val=0
    # attentions=model.get_last_selfattention(image.cuda())
    # attentions_all = model.forward_selfattention(image.cuda(),return_all_attention=True) # change here
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

        # th_attn_all.append(th_attn)

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

