# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
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
    'CrossImageVIT'
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
        load_model_path='domainbed/pretrained/single_train_models/DeitSmall/2f564cbde0093378552ac5c29dad5a0a/best_val_model_valdom_1_0.9701.pkl' #should not be the test model
        deit_trained_dgbed=load_model(load_model_path)
        self.network=MCVisionTransformer(img_size=224,num_classes=num_classes, distilled=False,patch_size=16, embed_dim=384, depth=12, num_heads=6,num_cls_emb=num_domains)
        self.network.load_state_dict(deit_trained_dgbed.network.state_dict(),strict=False) 
        printNetworkParams(self.network)
        # img = torch.randn(10, 3, 224, 224)
        # out=self.network(img)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # All teacher model paths. Domains should be in order ACPS
        teachers=['domainbed/pretrained/single_train_models/ERM/c52ccf0a2ff8fcb27e5f0c637efb5f90/best_val_model_valdom_0_0.9511.pkl','domainbed/pretrained/single_train_models/ERM/c62da0be82f89501d09e0a0e8fd65200/best_val_model_valdom_1_0.9744.pkl','domainbed/pretrained/single_train_models/ERM/cde183b2a672bd049e697054ef75e988/best_val_model_valdom_2_0.9940.pkl','domainbed/pretrained/single_train_models/ERM/6d29d7ebbbc85f0d744e924bb602d5ad/best_val_model_valdom_3_0.9720.pkl']
        test_envs=queue_var.current_test_env
        test_envs.sort()
        for _ in test_envs:
            del teachers[test_envs[-1]]  # Deleting test environment models
        assert len(teachers)==num_domains ,"Required number of teacher pretrained models not given"
        self.Teachnetwork=[load_model(fname) for fname in teachers]
    

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss=0
        predictions=torch.chunk(self.predictTrain(all_x),self.num_domains)

        t = 3.0
        for dom in range(self.num_domains):
            loss+= F.cross_entropy(predictions[0][:,dom], minibatches[dom][1])
            loss+= F.kl_div(
                F.log_softmax(predictions[0][:,dom] / t, dim=1),
                F.log_softmax(self.predictTeacher(self.Teachnetwork[dom],minibatches[dom][0]) / t, dim=1),
                reduction='sum',
                log_target=True
            ) * (t * t) / predictions[0][:,dom].numel()
    

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

       
    def predict(self, x):
        return torch.sum(self.network(x),1)
    def predictTrain(self, x):
        return self.network(x)
    def predictTeacher(self,net, x):
        return net(x)

class DeitSmall(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DeitSmall, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
    
        # self.network = torch.hub.load('/home/computervision1/Sanoojan/DomainBedS/deit',
        #                               'deit_small_patch16_224', pretrained=True, source='local')    
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
                    
        # self.network = torch.hub.load('/home/computervision1/Sanoojan/DomainBedS/deit',
        #                               'deit_Tiny_patch16_224', pretrained=True, source='local')    
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
                    
        # self.network = torch.hub.load('/home/computervision1/Sanoojan/DomainBedS/deit',
        #                               'deit_Tiny_patch16_224', pretrained=True, source='local')    
        self.network=tiny_cvt(pretrained=False) 
        # print(self.network)
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

class CrossImageVIT(Algorithm):
    """
    Empirical Risk Minimization with Deit (Deit-small)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CrossImageVIT, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.countersave=0   
        self.saveSamples=False       
        self.num_domains=num_domains

        self.network_deit=deit_small_patch16_224(pretrained=True) 
        self.network_deit.head = nn.Linear(384, num_classes)
        printNetworkParams(self.network_deit)
        self.network=CrossVisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=384, depth=1,
            im_enc_depth=12,cross_attn_depth=3,num_heads=6, representation_size=None, distilled=False,
            drop_rate=0., norm_layer=None, weight_init='',cross_attn_heads = 6,cross_attn_dim_head = 64,dropout = 0.1,im_enc_mlp_dim=1536,im_enc_dim_head=64,skipconnection=True)
        printNetworkParams(self.network)
        self.network.load_state_dict(self.network_deit.state_dict(),strict=False)
        self.network.blocks[0].load_state_dict(self.network_deit.state_dict(),strict=False)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
    def update(self, minibatches, unlabeled=None):

        train_queues = queue_var.train_queues
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
        cross_learning_data=[[] for i in range(ndomains)]  
        cross_learning_labels=[]
        for cls in range(nclass):
            for i in range(queue_sz) :
                for dom_n in range(ndomains):
                    cross_learning_data[dom_n].append(train_queues[cls][dom_n][i])
                cross_learning_labels.append(cls)

        cross_learning_data=[torch.stack(data) for data in cross_learning_data]
        cross_learning_labels=torch.tensor(cross_learning_labels).to("cuda")
        if (self.countersave<6 and self.saveSamples):
            for dom_n in range(ndomains):
                batch_images = make_grid(cross_learning_data[dom_n], nrow=queue_sz, normalize=True)
                save_image(batch_images, "./domainbed/image_outputs/batch_im_"+str(self.countersave)+"_"+str(dom_n)+".png",normalize=False)
            self.countersave+=1
        pred=self.predictTrain(cross_learning_data)
        loss = F.cross_entropy(pred, cross_learning_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    def predict(self, x):
        return self.network([x]*self.num_domains)
    def predictTrain(self, x):
        return self.network(x)


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