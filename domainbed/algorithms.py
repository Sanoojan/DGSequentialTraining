# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision
from vit_pytorch import ViT
# from timm.models import create_model
# from timm.models.vision_transformer import _cfg
from domainbed.lib.visiontransformer import *
from domainbed.lib.cross_visiontransformer import CrossVisionTransformer
from domainbed.lib.MCT import MCVisionTransformer
from domainbed.lib.cvt import tiny_cvt,small_cvt
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
    'MultiDomainDistillation'
    'DeitSmall',
    'DeitTiny',
    'CVTTiny',
    'CrossImageVIT',
    'DeitSmall_StrongTeachers',
    'DeitSmall_StrongTeachers_nodist'
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
        predictions=torch.chunk(self.predictTrain(all_x),self.num_domains,dim=0)

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


class DeitSmall(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitSmall, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
    
        self.network=deit_small_patch16_224(pretrained=True) 
        self.network.head = nn.Linear(384, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer for distillation
        
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

class DeitSmall_StrongTeachers(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitSmall_StrongTeachers, self).__init__(input_shape, num_classes, num_domains,
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
                      
        self.network=small_cvt(pretrained=False) 
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
    temp=0.1
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