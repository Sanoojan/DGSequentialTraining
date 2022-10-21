# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest.mock import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import copy
import numpy as np
import einops
import itertools
import domainbed.lib.clip.clip as clip
from collections import defaultdict, OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None
from domainbed.lib.visiontransformer import *
from domainbed import networks
from domainbed.lib.clip.clip import tokenize
from domainbed import global_var
from domainbed.lib import misc
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)


ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
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
        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        # print(self.network)
        printNetworkParams(self.network)
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

class ERM_ViT(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_ViT, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.network = networks.ViT(input_shape, self.hparams,num_classes)
        # self.classifier = networks.Classifier(768,num_classes,hparams['nonlinear_classifier'],init=hparams['weight_init']).to("cuda")
        # self.network=nn.Sequential(self.featurizer,self.classifier)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
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

class ERM_with_clip(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_with_clip, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.network=torchvision.models.resnet50(pretrained=True)
        del self.network.fc
        self.network.fc = nn.Identity()

        self.vis_proj = torch.nn.Linear(2048, 512)
        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_scratch"):
            print("clip_full")
            # self.featurizer.network.proj=None
        # else:

        #     self.featurizer.network.head=nn.Identity()

        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.network.parameters())+list(self.featurizer.visual.parameters())+list(self.vis_proj.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        self.num_domains=num_domains
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            self.text_features = self.featurizer.encode_text(text_inputs)
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        image_features = self.network(all_x)
        
        image_features = self.vis_proj(image_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()

        loss=F.cross_entropy(logits_per_image, all_y)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
       
        
        image_features = self.network(x)
    
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = self.vis_proj(image_features)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

class ERM_with_text_mix(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_with_text_mix, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.network=torchvision.models.resnet50(pretrained=True)
        del self.network.fc
        self.network.fc = nn.Identity()

        self.vis_proj = torch.nn.Linear(2048, 512)
        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_scratch"):
            print("clip_full")
            # self.featurizer.network.proj=None
        # else:

        #     self.featurizer.network.head=nn.Identity()

        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.network.parameters())+list(self.featurizer.visual.parameters())+list(self.vis_proj.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        self.num_domains=num_domains
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            self.text_features = self.featurizer.encode_text(text_inputs)
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        image_features = self.network(all_x)
        mixup_features=torch.clone(image_features)

        mixup_features_chunk=torch.chunk(mixup_features,chunks=self.num_domains)
        
        mixup_text_feature=torch.index_select(self.text_features, 0, all_y)
      

        mixup_text_chunk=torch.chunk(mixup_text_feature,chunks=self.num_domains)

        bs=int(len(all_x)/self.num_domains)
        
        a=torch.rand(int(len(all_x)),self.num_domains)
        sum=torch.sum(a,dim=1,keepdims=True)
        a=(a*(1)/sum).to("cuda")
        mixup_features=torch.unsqueeze(a[:,0],dim=1).expand(-1,2048)*mixup_features
        mixup_text_feature=torch.unsqueeze(a[:,0],dim=1).expand(-1,512)*mixup_text_feature
        for d in range(1,self.num_domains):
            rand_perm=torch.randperm(bs)
            mixup_features+=torch.unsqueeze(a[:,d],dim=1).expand(-1,2048)*torch.cat(([mixup_features_chunk[(dom+d)%self.num_domains][rand_perm] for dom in range(self.num_domains)]),dim=0)
            mixup_text_feature+=torch.unsqueeze(a[:,d],dim=1).expand(-1,512)*torch.cat(([mixup_text_chunk[(dom+d)%self.num_domains][rand_perm] for dom in range(self.num_domains)]),dim=0)
        # image_features=torch.cat((image_features,mixup_features),dim=0)
        # all_y_full=torch.cat((all_y_full,all_y),dim=0)

        
       
            # mixup_text_feature=self.featurizer.encode_text(mixup_text,no_embed=True,EOS_pos=EOS_pos)
        image_features = self.vis_proj(image_features)
        mixup_features= self.vis_proj(mixup_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        mixup_features = mixup_features / mixup_features.norm(dim=1, keepdim=True)

        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        mixup_text_feature = mixup_text_feature / mixup_text_feature.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image_mixup=logit_scale * mixup_features @ mixup_text_feature.t()
        logits_per_text_mixup = logits_per_image_mixup.t()
        
        labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image_mixup, labels)
        loss_t = F.cross_entropy(logits_per_text_mixup, labels)
        loss = (loss_i + loss_t)/2
    
        loss+=F.cross_entropy(logits_per_image, all_y)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
       
        
        image_features = self.network(x)
    
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = self.vis_proj(image_features)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

class ERM_ViT_mixup(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_ViT_mixup, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.network = networks.ViT(input_shape, self.hparams,num_classes).network
        # self.classifier = networks.Classifier(768,num_classes,hparams['nonlinear_classifier'],init=hparams['weight_init']).to("cuda")
        # self.network=nn.Sequential(self.featurizer,self.classifier)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        
        
        self.num_domains=num_domains
        self.mixup_weight=self.hparams['mixup_weight']
   
        print("mixup_weight:",self.mixup_weight)
 
        self.num_mixups=self.hparams['num_mixups']
        self.cascaded=self.hparams['cascaded']

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_y_full=torch.clone(all_y)
        
        image_features=self.network(all_x,ret_feat=True)
        mixup_features=torch.clone(image_features)
        mixup_features_chunk=torch.chunk(mixup_features,chunks=self.num_domains)
        bs=int(len(all_x)/self.num_domains)
        for rep in range(self.num_mixups):
            if (self.cascaded):
                mixup_features=self.mixup_weight*mixup_features
            else:
                mixup_features=self.mixup_weight*image_features
            a=torch.rand(int(len(all_x)),self.num_domains-1)
            sum=torch.sum(a,dim=1,keepdims=True)
            a=(a*(1-self.mixup_weight)/sum).to("cuda")
            for d in range(self.num_domains-1):
                mixup_features+=torch.unsqueeze(a[:,d],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[(dom+d+1)%self.num_domains][torch.randperm(bs)] for dom in range(self.num_domains)]),dim=0)
            image_features=torch.cat((image_features,mixup_features),dim=0)
            all_y_full=torch.cat((all_y_full,all_y),dim=0)
        
        preds=self.network.head(image_features)
        loss = F.cross_entropy(preds, all_y_full)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class ERM_ViT_Clip_train(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_ViT_Clip_train, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.network=networks.ViT(input_shape, self.hparams,num_classes).network
        self.network.head=nn.Identity()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/16', device)
        # model=model.float()
        self.featurizer=model.float()

        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.network.parameters())+list(self.featurizer.visual.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            self.text_features = self.featurizer.encode_text(text_inputs)
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        image_features = self.network(all_x,ret_feat=True)
        image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        
        
        image_features = self.network(x,ret_feat=True)
        
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
    
        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

class ERM_ViT_with_text_mix(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_ViT_with_text_mix, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.network=networks.ViT(input_shape, self.hparams,num_classes).network
        self.network.head=nn.Identity()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/16', device)
        # model=model.float()
        self.featurizer=model.float()

        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.network.parameters())+list(self.featurizer.visual.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        self.num_domains=num_domains
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            self.text_features = self.featurizer.encode_text(text_inputs)
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        image_features = self.network(all_x,ret_feat=True)
        mixup_features=torch.clone(image_features)

        mixup_features_chunk=torch.chunk(mixup_features,chunks=self.num_domains)
        
        mixup_text_feature=torch.index_select(self.text_features, 0, all_y)
      

        mixup_text_chunk=torch.chunk(mixup_text_feature,chunks=self.num_domains)

        bs=int(len(all_x)/self.num_domains)
        
        a=torch.rand(int(len(all_x)),self.num_domains)
        sum=torch.sum(a,dim=1,keepdims=True)
        a=(a*(1)/sum).to("cuda")
        mixup_features=torch.unsqueeze(a[:,0],dim=1).expand(-1,768)*mixup_features
        mixup_text_feature=torch.unsqueeze(a[:,0],dim=1).expand(-1,512)*mixup_text_feature
        for d in range(1,self.num_domains):
            rand_perm=torch.randperm(bs)
            mixup_features+=torch.unsqueeze(a[:,d],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[(dom+d)%self.num_domains][rand_perm] for dom in range(self.num_domains)]),dim=0)
            mixup_text_feature+=torch.unsqueeze(a[:,d],dim=1).expand(-1,512)*torch.cat(([mixup_text_chunk[(dom+d)%self.num_domains][rand_perm] for dom in range(self.num_domains)]),dim=0)
        # image_features=torch.cat((image_features,mixup_features),dim=0)
        # all_y_full=torch.cat((all_y_full,all_y),dim=0)

        
       
            # mixup_text_feature=self.featurizer.encode_text(mixup_text,no_embed=True,EOS_pos=EOS_pos)
        image_features = image_features @ self.featurizer.visual.proj
        mixup_features=mixup_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        mixup_features = mixup_features / mixup_features.norm(dim=1, keepdim=True)

        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        mixup_text_feature = mixup_text_feature / mixup_text_feature.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image_mixup=logit_scale * mixup_features @ mixup_text_feature.t()
        logits_per_text_mixup = logits_per_image_mixup.t()
        
        labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image_mixup, labels)
        loss_t = F.cross_entropy(logits_per_text_mixup, labels)
        loss = (loss_i + loss_t)/2
    
        loss+=F.cross_entropy(logits_per_image, all_y)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
       
        
        image_features = self.network(x,ret_feat=True)
    
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

class ERM_ViT_classifier_learning(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_ViT_classifier_learning, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes)
        if(self.hparams['weight_init']=="clip"):
            self.featurizer.network.proj=None
        else:

            self.featurizer.network.head=nn.Identity()
        self.classifier = networks.Classifier(
            768,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
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

class ERM_ViT_LPFT(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_ViT_LPFT, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes)
        if(self.hparams['weight_init']=="clip"):
            self.featurizer.network.proj=None
        else:

            self.featurizer.network.head=nn.Identity()
        self.classifier = networks.Classifier(
            768,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        if(self.cnt<2500):
            with torch.no_grad():
                feat=self.featurizer(all_x)
            loss=F.cross_entropy(self.classifier(feat), all_y)
        else:
            loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class ERM_clip_text_conc_Frz(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_text_conc_Frz, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.featurizer_orig = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        else:

            self.featurizer.network.head=nn.Identity()
        self.classifier = networks.Classifier(
            1280,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
            image_features = self.featurizer.encode_image(all_x)
            text_features = self.featurizer.encode_text(text_inputs)
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
        with torch.no_grad():
            image_features = self.featurizer_orig.encode_image(x)
            text_features = self.featurizer_orig.encode_text(text_inputs)
            # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
            image_features = image_features @ self.featurizer_orig.visual.proj

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        

            # cosine similarity as logits
            logit_scale = self.featurizer_orig.logit_scale.exp()

            logits_per_image = logit_scale * image_features @ text_features.t()
            indices=torch.argmax(logits_per_image,dim=1)
            text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in indices]).to("cuda")
            image_features = self.featurizer.encode_image(x)
            text_features = self.featurizer.encode_text(text_inputs)
            conc_feat=torch.cat([image_features,text_features],dim=1)
            outs=self.classifier(conc_feat)
            # print(indices)

        return outs

class ERM_clip_text_conc(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_text_conc, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.featurizer_orig = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        else:

            self.featurizer.network.head=nn.Identity()
        self.classifier = networks.Classifier(
            1280,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        image_features = self.featurizer.encode_image(all_x)
        text_features = self.featurizer.encode_text(text_inputs)
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
        indices=torch.argmax(logits_per_image,dim=1)
        text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in indices]).to("cuda")
        image_features = self.featurizer.encode_image(x)
        text_features = self.featurizer.encode_text(text_inputs)
        conc_feat=torch.cat([image_features,text_features],dim=1)
        outs=self.classifier(conc_feat)
        return outs



class Clip_train(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        # else:

        #     self.featurizer.network.head=nn.Identity()

        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
        image_features = self.featurizer.encode_image(all_x)
        text_features = self.featurizer.encode_text(text_inputs)
        image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        text_inputs = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
        
        image_features = self.featurizer.encode_image(x)
        text_features = self.featurizer.encode_text(text_inputs)
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

class Clip_train_text_freeze(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train_text_freeze, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None

        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            self.text_features = self.featurizer.encode_text(text_inputs)
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
       
        image_features = self.featurizer.encode_image(all_x)
      
        image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
 
        
        image_features = self.featurizer.encode_image(x)
        
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

class Clip_train_prompt(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train_prompt, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        # else:

        #     self.featurizer.network.head=nn.Identity()

        # self.network = nn.Sequential(self.featurizer, self.classifier)
        num_prompts=10
        # self.prompts = nn.Parameter(torch.zeros(1, num_prompts, 512))
        # self.prompts.requires_grad_()
        # nn.init.trunc_normal_(self.prompts, std=.02)

        for name,param in self.featurizer.transformer.named_parameters():
            if name!="prompts":
                # print(name)
                param.requires_grad = False
        

        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0
        self.num_classes=num_classes
        prompt_prefix = ' '.join(['X'] * num_prompts)
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.Class_names]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to("cuda")
        with torch.no_grad():
            embedding = self.featurizer.token_embedding(self.tokenized_prompts).type(self.featurizer.dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        self.register_buffer('token_suffix', embedding[:, num_prompts + 1:, :])  # CLS, EOS

    def _get_text_features(self, coop=False):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]

        # domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)

        # prompt_feature=self.prompts.expand(x.shape[0], -1, -1)
        prompt_feature=self.featurizer.prompts.expand(self.num_classes, -1, -1)
        
        # print(self.token_prefix.shape)
        # print(prompt_feature.shape)
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        prompt_feature = torch.cat([self.token_prefix, prompt_feature, self.token_suffix], dim=1)
        
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = prompt_feature + self.featurizer.positional_embedding.type(self.featurizer.dtype)
        x = x.permute(1, 0, 2)
        x = self.featurizer.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.featurizer.ln_final(x).type(self.featurizer.dtype)
        
        #  mapping domain_features to text_features.
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.featurizer.text_projection      
        # print(text_features.shape)
        return text_features

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        # text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
        image_features = self.featurizer.encode_image(all_x)

        text_features = self._get_text_features()
        image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        text_inputs = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
        
        image_features = self.featurizer.encode_image(x)
        # text_features = self.featurizer.encode_text(text_inputs)
        text_features = self._get_text_features()
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

class Clip_train_mixup(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train_mixup, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        # else:

        #     self.featurizer.network.head=nn.Identity()

        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0
        self.num_domains=num_domains
        self.mixup_weight=0.6

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        image_features = self.featurizer.encode_image(all_x)
        mixup_features=torch.clone(image_features)
        mixup_features_chunk=torch.chunk(mixup_features,chunks=self.num_domains)
        mixup_features=self.mixup_weight*mixup_features
        
        a=torch.rand(int(len(all_x)),self.num_domains-1)
        sum=torch.sum(a,dim=1,keepdims=True)
        a=(a*(1-self.mixup_weight)/sum).to("cuda")

        #for now works only for 4 domain datasets
        if(self.num_domains==3):
            mixup_features2=torch.unsqueeze(a[:,0],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[1],mixup_features_chunk[2],mixup_features_chunk[0]]),dim=0)
            mixup_features3=torch.unsqueeze(a[:,1],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[2],mixup_features_chunk[0],mixup_features_chunk[1]]),dim=0)
        elif(self.num_domains==5):
            mixup_features2=torch.unsqueeze(a[:,0],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[1],mixup_features_chunk[2],mixup_features_chunk[3],mixup_features_chunk[4],mixup_features_chunk[0]]),dim=0)
            mixup_features3=torch.unsqueeze(a[:,1],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[2],mixup_features_chunk[3],mixup_features_chunk[4],mixup_features_chunk[0],mixup_features_chunk[1]]),dim=0)
            mixup_features4=torch.unsqueeze(a[:,2],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[3],mixup_features_chunk[4],mixup_features_chunk[0],mixup_features_chunk[1],mixup_features_chunk[2]]),dim=0)
            mixup_features5=torch.unsqueeze(a[:,3],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[4],mixup_features_chunk[0],mixup_features_chunk[1],mixup_features_chunk[2],mixup_features_chunk[3]]),dim=0)
       

        if(self.num_domains==3):
            mixup_features=mixup_features+mixup_features2+mixup_features3
        elif(self.num_domains==5):
            mixup_features=mixup_features+mixup_features2+mixup_features3+mixup_features4+mixup_features5
       
 
        
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            text_features = self.featurizer.encode_text(text_inputs)
        image_features = image_features @ self.featurizer.visual.proj
        mixup_features=mixup_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        mixup_features = mixup_features / mixup_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image_mixup=logit_scale * mixup_features @ text_features.t()
    
        loss=F.cross_entropy(logits_per_image, all_y)
        loss1=F.cross_entropy(logits_per_image_mixup, all_y)
        loss=loss+loss1
        
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        text_inputs = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
        
        image_features = self.featurizer.encode_image(x)
        text_features = self.featurizer.encode_text(text_inputs)
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

class Clip_train_mixup_with_text(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train_mixup_with_text, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
       
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            self.text_features = self.featurizer.encode_text(text_inputs)
        self.cnt=0
        self.num_domains=num_domains
        # self.mixup_weight=0.6

    
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # print(all_y)
        image_features = self.featurizer.encode_image(all_x)
        mixup_features=torch.clone(image_features)

        mixup_features_chunk=torch.chunk(mixup_features,chunks=self.num_domains)
        
        mixup_text_feature=torch.index_select(self.text_features, 0, all_y)
        # EOS_pos=mixup_text.argmax(dim=-1)
        # mixup_text=self.featurizer.token_embedding(mixup_text).type(self.featurizer.dtype)
        mixup_text_chunk=torch.chunk(mixup_text_feature,chunks=self.num_domains)

        bs=int(len(all_x)/self.num_domains)
        ba=int(len(all_x))
        a=torch.rand(int(len(all_x)),self.num_domains)
        sum=torch.sum(a,dim=1,keepdims=True)
        a=(a*(1)/sum).to("cuda")
        mixup_features=torch.unsqueeze(a[:,0],dim=1).expand(-1,768)*mixup_features
        mixup_text_feature=torch.unsqueeze(a[:,0],dim=1).expand(-1,512)*mixup_text_feature
        for d in range(1,self.num_domains):
            mixup_features+=torch.unsqueeze(a[:,d],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[(dom+d)%self.num_domains] for dom in range(self.num_domains)]),dim=0)
            mixup_text_feature+=torch.unsqueeze(a[:,d],dim=1).expand(-1,512)*torch.cat(([mixup_text_chunk[(dom+d)%self.num_domains] for dom in range(self.num_domains)]),dim=0)
  
        
        text_features = self.text_features
        image_features = image_features @ self.featurizer.visual.proj
        mixup_features=mixup_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        mixup_features = mixup_features / mixup_features.norm(dim=1, keepdim=True)

        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        mixup_text_feature = mixup_text_feature / mixup_text_feature.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image_mixup=logit_scale * mixup_features @ mixup_text_feature.t()
        logits_per_text_mixup = logits_per_image_mixup.t()
        
        labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image_mixup, labels)
        loss_t = F.cross_entropy(logits_per_text_mixup, labels)
        loss = (loss_i + loss_t)/2.0
    
        loss+=F.cross_entropy(logits_per_image, all_y)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}


    def predict(self, x):

        image_features = self.featurizer.encode_image(x)
        text_features = self.text_features
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

class Clip_train_mixup_with_text_hetro(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train_mixup_with_text_hetro, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
       
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Train_class_names=misc.Train_class_names
        self.Class_names=misc.Class_names
        self.class_change=misc.class_change.to("cuda")
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            self.text_features = self.featurizer.encode_text(text_inputs)
        with torch.no_grad():
            text_inputs_train  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Train_class_names]).to("cuda")
            self.train_text_features = self.featurizer.encode_text(text_inputs_train)
        self.cnt=0
        self.num_domains=num_domains
        # self.mixup_weight=0.6

    
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_y=torch.index_select(self.class_change,0,all_y)
        # print(all_y)
        image_features = self.featurizer.encode_image(all_x)
        mixup_features=torch.clone(image_features)

        mixup_features_chunk=torch.chunk(mixup_features,chunks=self.num_domains)
        
        mixup_text_feature=torch.index_select(self.train_text_features, 0, all_y)
        # EOS_pos=mixup_text.argmax(dim=-1)
        # mixup_text=self.featurizer.token_embedding(mixup_text).type(self.featurizer.dtype)
        mixup_text_chunk=torch.chunk(mixup_text_feature,chunks=self.num_domains)

        bs=int(len(all_x)/self.num_domains)
        ba=int(len(all_x))
        a=torch.rand(int(len(all_x)),self.num_domains)
        sum=torch.sum(a,dim=1,keepdims=True)
        a=(a*(1)/sum).to("cuda")
        mixup_features=torch.unsqueeze(a[:,0],dim=1).expand(-1,768)*mixup_features
        mixup_text_feature=torch.unsqueeze(a[:,0],dim=1).expand(-1,512)*mixup_text_feature
        for d in range(1,self.num_domains):
            mixup_features+=torch.unsqueeze(a[:,d],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[(dom+d)%self.num_domains] for dom in range(self.num_domains)]),dim=0)
            mixup_text_feature+=torch.unsqueeze(a[:,d],dim=1).expand(-1,512)*torch.cat(([mixup_text_chunk[(dom+d)%self.num_domains] for dom in range(self.num_domains)]),dim=0)
  
        
        text_features = self.train_text_features
        image_features = image_features @ self.featurizer.visual.proj
        mixup_features=mixup_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        mixup_features = mixup_features / mixup_features.norm(dim=1, keepdim=True)

        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        mixup_text_feature = mixup_text_feature / mixup_text_feature.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image_mixup=logit_scale * mixup_features @ mixup_text_feature.t()
        logits_per_text_mixup = logits_per_image_mixup.t()
        
        labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image_mixup, labels)
        loss_t = F.cross_entropy(logits_per_text_mixup, labels)
        loss = (loss_i + loss_t)/2.0
    
        # loss=F.cross_entropy(logits_per_image, all_y)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}


    def predict(self, x):

        image_features = self.featurizer.encode_image(x)
        text_features = self.text_features
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image


class zero_shot_eval(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(zero_shot_eval, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        fname="domainbed/outputs_clip/Ablation/ERM_ViT_with_text_mix-DeitBase/OfficeHome/lr-0.00005/8c4057b843744d2fa58adad4168669db/best_val_model_testdom_[0]_0.8869.pkl"
        
        model=load_model(fname)

        self.network = model.network
        print(type(self.network).__name__)
        self.featurizer=model.featurizer
        print("feat:",type(self.featurizer).__name__)
        printNetworkParams(self.featurizer)
 
        self.Class_names=misc.Class_names
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            self.text_features = self.featurizer.encode_text(text_inputs)
        self.cnt=0
        self.num_domains=num_domains
        # self.mixup_weight=0.6

    def update(self, minibatches, unlabeled=None):
        return None

    def predict(self, x):
       
        
        image_features = self.network(x,ret_feat=True)
    
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image

# class zero_shot_eval(Algorithm):
#     """
#     Empirical Risk Minimization (ERM)
#     """

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(zero_shot_eval, self).__init__(input_shape, num_classes, num_domains,
#                                   hparams)
#         fname="domainbed/outputs_clip/Deitbase_related_ablations/Deitbase_with_text_mix/PACS/lr-0.00005/821efa20d1443f38d14601edbd239c9e/best_val_model_testdom_[2]_0.9768.pkl"
        
#         model=load_model(fname)

#         self.featurizer = model.featurizer
        
#         printNetworkParams(self.featurizer)
 
#         self.Class_names=misc.Class_names
#         with torch.no_grad():
#             text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
#             self.text_features = self.featurizer.encode_text(text_inputs)
#         self.cnt=0
#         self.num_domains=num_domains
#         # self.mixup_weight=0.6

#     def update(self, minibatches, unlabeled=None):
#         return None

#     def predict(self, x):

#         image_features = self.featurizer.encode_image(x)
#         text_features = self.text_features
#         image_features = image_features @ self.featurizer.visual.proj

#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

#         # cosine similarity as logits
#         logit_scale = self.featurizer.logit_scale.exp()

#         logits_per_image = logit_scale * image_features @ text_features.t()
        
#         return logits_per_image

class Clip_train_mixup_with_text_i(Clip_train_mixup_with_text):
    """
    Empirical Risk Minimization (ERM)
    """

    
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        image_features = self.featurizer.encode_image(all_x)
        mixup_features=torch.clone(image_features)

        mixup_features_chunk=torch.chunk(mixup_features,chunks=self.num_domains)
        
        mixup_text_feature=torch.index_select(self.text_features, 0, all_y)
        # EOS_pos=mixup_text.argmax(dim=-1)
        # mixup_text=self.featurizer.token_embedding(mixup_text).type(self.featurizer.dtype)
        mixup_text_chunk=torch.chunk(mixup_text_feature,chunks=self.num_domains)

        bs=int(len(all_x)/self.num_domains)
        ba=int(len(all_x))
        a=torch.rand(int(len(all_x)),self.num_domains)
        sum=torch.sum(a,dim=1,keepdims=True)
        a=(a*(1)/sum).to("cuda")
        rand_perm=torch.randperm(ba)
        mixup_features=torch.unsqueeze(a[:,0],dim=1).expand(-1,768)*mixup_features[rand_perm]
        mixup_text_feature=torch.unsqueeze(a[:,0],dim=1).expand(-1,512)*mixup_text_feature[rand_perm]
        for d in range(1,self.num_domains):
            rand_perm=torch.randperm(bs)
            mixup_features+=torch.unsqueeze(a[:,d],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[(dom+d)%self.num_domains][rand_perm] for dom in range(self.num_domains)]),dim=0)
            mixup_text_feature+=torch.unsqueeze(a[:,d],dim=1).expand(-1,512)*torch.cat(([mixup_text_chunk[(dom+d)%self.num_domains][rand_perm] for dom in range(self.num_domains)]),dim=0)
  
        
        text_features = self.text_features
        image_features = image_features @ self.featurizer.visual.proj
        mixup_features=mixup_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        mixup_features = mixup_features / mixup_features.norm(dim=1, keepdim=True)

        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        mixup_text_feature = mixup_text_feature / mixup_text_feature.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image_mixup=logit_scale * mixup_features @ mixup_text_feature.t()
        logits_per_text_mixup = logits_per_image_mixup.t()
        
        labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image_mixup, labels)
        
        loss = loss_i 
    
        loss+=F.cross_entropy(logits_per_image, all_y)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}




class Clip_train_mixup_with_text_cls(Clip_train_mixup):
    """
    Empirical Risk Minimization (ERM)
    """
    
    
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        image_features = self.featurizer.encode_image(all_x)
        mixup_features=torch.clone(image_features)

        # mixup_features_chunk=torch.chunk(mixup_features,chunks=self.num_domains)
        
        mixup_text=torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        with torch.no_grad():
            mixup_text_feature =self.featurizer.encode_text(mixup_text)
        # EOS_pos=mixup_text.argmax(dim=-1)
        # mixup_text=self.featurizer.token_embedding(mixup_text).type(self.featurizer.dtype)
        # mixup_text_chunk=torch.chunk(mixup_text_feature,chunks=self.num_domains)

        bs=int(len(all_x))
        
        a=torch.rand(int(len(all_x)),2)
        sum=torch.sum(a,dim=1,keepdims=True)
        a=(a*(1)/sum).to("cuda")
        mixup_features=torch.unsqueeze(a[:,0],dim=1).expand(-1,768)*mixup_features
        mixup_text_feature=torch.unsqueeze(a[:,0],dim=1).expand(-1,512)*mixup_text_feature
        for d in range(1,2):
            rand_perm=torch.randperm(bs)
            mixup_features+=torch.unsqueeze(a[:,d],dim=1).expand(-1,768)*mixup_features[rand_perm]
            mixup_text_feature+=torch.unsqueeze(a[:,d],dim=1).expand(-1,512)*mixup_text_feature[rand_perm]
        # image_features=torch.cat((image_features,mixup_features),dim=0)
        # all_y_full=torch.cat((all_y_full,all_y),dim=0)

        
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            text_features = self.featurizer.encode_text(text_inputs)
            # mixup_text_feature=self.featurizer.encode_text(mixup_text,no_embed=True,EOS_pos=EOS_pos)
        image_features = image_features @ self.featurizer.visual.proj
        mixup_features=mixup_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        mixup_features = mixup_features / mixup_features.norm(dim=1, keepdim=True)

        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        mixup_text_feature = mixup_text_feature / mixup_text_feature.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image_mixup=logit_scale * mixup_features @ mixup_text_feature.t()
        logits_per_text_mixup = logits_per_image_mixup.t()
        
        labels = torch.tensor(np.arange(len(all_x))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image_mixup, labels)
        loss_t = F.cross_entropy(logits_per_text_mixup, labels)
        loss = (loss_i + loss_t)/2
    
        loss+=F.cross_entropy(logits_per_image, all_y)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

   

class Clip_domain_mixup(Clip_train_mixup):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_domain_mixup, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.mixup_weight=self.hparams['mixup_weight']
   
        print("mixup_weight:",self.mixup_weight)
 
        self.num_mixups=self.hparams['num_mixups']
        self.cascaded=self.hparams['cascaded']

    def update(self, minibatches, unlabeled=None):
    
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_y_full=torch.clone(all_y)
        image_features = self.featurizer.encode_image(all_x)
        mixup_features=torch.clone(image_features)
        mixup_features_chunk=torch.chunk(mixup_features,chunks=self.num_domains)
        

        bs=int(len(all_x)/self.num_domains)
        for rep in range(self.num_mixups):
            if(self.cascaded):
                mixup_features=self.mixup_weight*mixup_features
            else:
                mixup_features=self.mixup_weight*image_features
            a=torch.rand(int(len(all_x)),self.num_domains-1)
            sum=torch.sum(a,dim=1,keepdims=True)
            a=(a*(1-self.mixup_weight)/sum).to("cuda")
            for d in range(self.num_domains-1):
                mixup_features+=torch.unsqueeze(a[:,d],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[(dom+d+1)%self.num_domains][torch.randperm(bs)] for dom in range(self.num_domains)]),dim=0)
            image_features=torch.cat((image_features,mixup_features),dim=0)
            all_y_full=torch.cat((all_y_full,all_y),dim=0)
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            text_features = self.featurizer.encode_text(text_inputs)
        image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y_full)
    
        
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

class Clip_domain_mixup_with_text_cascaded(Clip_train_mixup):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_domain_mixup_with_text_cascaded, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
       
        self.num_mixups=self.hparams['num_mixups']
        self.cascaded=self.hparams['cascaded']
       

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        image_features = self.featurizer.encode_image(all_x)
        mixup_feat=torch.clone(image_features)

        mixup_features_chunk=torch.chunk(mixup_feat,chunks=self.num_domains)
        
        mixup_text=torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        with torch.no_grad():
            mixup_text_feat =self.featurizer.encode_text(mixup_text)
            mixup_text_feat_ori=mixup_text_feat.clone()
        # EOS_pos=mixup_text.argmax(dim=-1)
        # mixup_text=self.featurizer.token_embedding(mixup_text).type(self.featurizer.dtype)
        mixup_text_chunk=torch.chunk(mixup_text_feat,chunks=self.num_domains)

        bs=int(len(all_x)/self.num_domains)
        mixup_features=torch.Tensor().to("cuda")
        mixup_text_feature=torch.Tensor().to("cuda")
        for rep in range(self.num_mixups): 
            a=torch.rand(int(len(all_x)),self.num_domains)
            sum=torch.sum(a,dim=1,keepdims=True)
            a=(a*(1)/sum).to("cuda")
            if(self.cascaded):
                mixup_feat=torch.unsqueeze(a[:,0],dim=1).expand(-1,768)*mixup_feat
                mixup_text_feat=torch.unsqueeze(a[:,0],dim=1).expand(-1,512)*mixup_text_feat
            else:
                mixup_feat=torch.unsqueeze(a[:,0],dim=1).expand(-1,768)*image_features
                mixup_text_feat=torch.unsqueeze(a[:,0],dim=1).expand(-1,512)*mixup_text_feat_ori
            for d in range(1,self.num_domains):
                rand_perm=torch.randperm(bs)
                mixup_feat+=torch.unsqueeze(a[:,d],dim=1).expand(-1,768)*torch.cat(([mixup_features_chunk[(dom+d)%self.num_domains][rand_perm] for dom in range(self.num_domains)]),dim=0)
                mixup_text_feat+=torch.unsqueeze(a[:,d],dim=1).expand(-1,512)*torch.cat(([mixup_text_chunk[(dom+d)%self.num_domains][rand_perm] for dom in range(self.num_domains)]),dim=0)
            mixup_features=torch.cat((mixup_features,mixup_feat),dim=0)
            mixup_text_feature=torch.cat((mixup_text_feature,mixup_text_feat),dim=0)

        
        with torch.no_grad():
            text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            text_features = self.featurizer.encode_text(text_inputs)
            # mixup_text_feature=self.featurizer.encode_text(mixup_text,no_embed=True,EOS_pos=EOS_pos)
        image_features = image_features @ self.featurizer.visual.proj
        mixup_features=mixup_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        mixup_features = mixup_features / mixup_features.norm(dim=1, keepdim=True)

        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        mixup_text_feature = mixup_text_feature / mixup_text_feature.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image_mixup=logit_scale * mixup_features @ mixup_text_feature.t()
        logits_per_text_mixup = logits_per_image_mixup.t()
        
        labels = torch.tensor(np.arange(len(logits_per_image_mixup))).to("cuda")

        loss_i = F.cross_entropy(logits_per_image_mixup, labels)
        loss_t = F.cross_entropy(logits_per_text_mixup, labels)
        loss = (loss_i + loss_t)/2
    
        loss+=F.cross_entropy(logits_per_image, all_y)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}



class ERM_clip_cross_attn(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_cross_attn, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        else:
            self.featurizer.network.head=nn.Identity()

        self.EMBEDDING_DIM=512
        self.prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        
        self.Class_names=misc.Class_names
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.Class_names]
        
        prompts = [self.prompt_prefix + ' ' + name + '.' for name in classnames]

        # print(prompts)
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        
        with torch.no_grad():
            embedding = self.featurizer.token_embedding(self.tokenized_prompts).type(self.featurizer.dtype)
            # print("embedding shape:",embedding.shape)
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS

        self.network = networks.MLP(768, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.featurizer.dtype)

        self.classifier = networks.Classifier(
            512,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters())+list(self.network.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        
        # print(self.Class_names)
        self.cnt=0

    def encode_text_with_image(self, text,image):
        image = image.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
        text_embedding = self.featurizer.token_embedding(text.to("cuda"))
        token_prefix=text_embedding[:, :1, :]
        token_suffix=text_embedding[:, self.hparams['num_domain_tokens'] + 1:, :]
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        # print(token_prefix.shape)
        # print(token_suffix.shape)
        # print(image.shape)
        image = torch.cat([token_prefix, image, token_suffix], dim=1)
        
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = image + self.featurizer.positional_embedding.type(self.featurizer.dtype)
        x = x.permute(1, 0, 2)
        x = self.featurizer.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.featurizer.ln_final(x).type(self.featurizer.dtype)
        # print(x.shape)
        #  mapping domain_features to text_features.
        # print(self.tokenized_prompts.argmax(dim=-1))
        
        text_features = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.featurizer.text_projection      # 0th token??
        return text_features

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        

        return x

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {self.Class_names[c]}") for c in all_y])
        image_features = self.featurizer.encode_image(all_x) 
        mlp_img_feat=self.network(image_features)
        fused_feat=self.encode_text_with_image(text_inputs,mlp_img_feat)
        # print(fused_feat.shape)
        text_features=self.featurizer.encode_text(torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {c}") for c in self.Class_names]).to("cuda"))
        image_features=image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        # conc_feat=torch.cat([image_features,text_features],dim=1)
        loss+=F.cross_entropy(self.classifier(fused_feat), all_y)
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        text_inputs = torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {c}") for c in self.Class_names]).to("cuda")
        
        image_features_im = self.featurizer.encode_image(x) 
        text_features = self.featurizer.encode_text(text_inputs)
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        

        image_features=image_features_im @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()
        

        logits_per_image = logit_scale * image_features @ text_features.t()
        prob=F.softmax(logits_per_image,dim=1)
        prob=torch.max(prob,dim=1)
        vals=prob.values.unsqueeze(1)
        indices=prob.indices

        text_inputs  = torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {self.Class_names[c]}") for c in indices])
        image_features = self.network(image_features_im)
        # text_embedding = self.featurizer.token_embedding(text_inputs)
        fused_feat=self.encode_text_with_image(text_inputs,image_features)
        
        outs=self.classifier(fused_feat)
        return outs

class ERM_clip_cross_attn_deitBase(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_cross_attn_deitBase, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        # else:
        #     self.featurizer.network.head=nn.Identity()

        self.EMBEDDING_DIM=512
        self.prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        
        self.Class_names=misc.Class_names
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.Class_names]
        
        prompts = [self.prompt_prefix + ' ' + name + '.' for name in classnames]

        # print(prompts)
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        
        with torch.no_grad():
            embedding = self.featurizer.token_embedding(self.tokenized_prompts).type(self.featurizer.dtype)
            # print("embedding shape:",embedding.shape)
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS

        self.network = networks.MLP(768, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.featurizer.dtype)

        self.classifier = networks.Classifier(
            512,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters())+list(self.network.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        
        # print(self.Class_names)
        self.cnt=0

    def encode_text_with_image(self, text,image):
        image = image.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
        text_embedding = self.featurizer.token_embedding(text.to("cuda"))
        token_prefix=text_embedding[:, :1, :]
        token_suffix=text_embedding[:, self.hparams['num_domain_tokens'] + 1:, :]
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        # print(token_prefix.shape)
        # print(token_suffix.shape)
        # print(image.shape)
        image = torch.cat([token_prefix, image, token_suffix], dim=1)
        
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = image + self.featurizer.positional_embedding.type(self.featurizer.dtype)
        x = x.permute(1, 0, 2)
        x = self.featurizer.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.featurizer.ln_final(x).type(self.featurizer.dtype)
        # print(x.shape)
        #  mapping domain_features to text_features.
        # print(self.tokenized_prompts.argmax(dim=-1))
        
        text_features = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.featurizer.text_projection      # 0th token??
        return text_features

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        

        return x

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {self.Class_names[c]}") for c in all_y])
        image_features = self.featurizer.encode_image(all_x) 
        mlp_img_feat=self.network(image_features)
        fused_feat=self.encode_text_with_image(text_inputs,mlp_img_feat)
        # print(fused_feat.shape)
        text_features=self.featurizer.encode_text(torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {c}") for c in self.Class_names]).to("cuda"))
        image_features=image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        # conc_feat=torch.cat([image_features,text_features],dim=1)
        loss+=F.cross_entropy(self.classifier(fused_feat), all_y)
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        text_inputs = torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {c}") for c in self.Class_names]).to("cuda")
        
        image_features_im = self.featurizer.encode_image(x) 
        text_features = self.featurizer.encode_text(text_inputs)
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        

        image_features=image_features_im @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()
        

        logits_per_image = logit_scale * image_features @ text_features.t()
        prob=F.softmax(logits_per_image,dim=1)
        prob=torch.max(prob,dim=1)
        vals=prob.values.unsqueeze(1)
        indices=prob.indices

        text_inputs  = torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {self.Class_names[c]}") for c in indices])
        image_features = self.network(image_features_im)
        # text_embedding = self.featurizer.token_embedding(text_inputs)
        fused_feat=self.encode_text_with_image(text_inputs,image_features)
        
        outs=self.classifier(fused_feat)
        return outs

class ERM_clip_featmatch(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_featmatch, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.featurizer_orig = networks.ViT(input_shape, self.hparams,num_classes).network
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        else:
            self.featurizer.network.head=nn.Identity()

        self.EMBEDDING_DIM=512
        self.prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        
        self.Class_names=misc.Class_names
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.Class_names]
        
        prompts = [self.prompt_prefix + ' ' + name + '.' for name in classnames]

        # print(prompts)
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        
        with torch.no_grad():
            embedding = self.featurizer.token_embedding(self.tokenized_prompts).type(self.featurizer.dtype)
            # print("embedding shape:",embedding.shape)
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS

        self.network = networks.MLP(768, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.featurizer.dtype)

        self.classifier = networks.Classifier(
            512,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters())+list(self.network.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        
        # print(self.Class_names)
        self.cnt=0

    def encode_text_with_image(self, text,image):
        image = image.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
        text_embedding = self.featurizer.token_embedding(text.to("cuda"))
        token_prefix=text_embedding[:, :1, :]
        token_suffix=text_embedding[:, self.hparams['num_domain_tokens'] + 1:, :]
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        # print(token_prefix.shape)
        # print(token_suffix.shape)
        # print(image.shape)
        image = torch.cat([token_prefix, image, token_suffix], dim=1)
        
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = image + self.featurizer.positional_embedding.type(self.featurizer.dtype)
        x = x.permute(1, 0, 2)
        x = self.featurizer.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.featurizer.ln_final(x).type(self.featurizer.dtype)
        # print(x.shape)
        #  mapping domain_features to text_features.
        # print(self.tokenized_prompts.argmax(dim=-1))
        
        text_features = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.featurizer.text_projection      # 0th token??
        return text_features

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        

        return x

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {self.Class_names[c]}") for c in all_y])
        image_features = self.network(self.featurizer.encode_image(all_x))
        
    
        fused_feat=self.encode_text_with_image(text_inputs,image_features)
        # print(fused_feat.shape)

        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # conc_feat=torch.cat([image_features,text_features],dim=1)
        loss=F.cross_entropy(self.classifier(fused_feat), all_y)
        
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
        
        image_features_im = self.featurizer_orig.encode_image(x)
        text_features = self.featurizer_orig.encode_text(text_inputs)
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features_im @ self.featurizer.visual.proj

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        # cosine similarity as logits
        logit_scale = self.featurizer_orig.logit_scale.exp()
        

        logits_per_image = logit_scale * image_features @ text_features.t()
        prob=F.softmax(logits_per_image,dim=1)
        prob=torch.max(prob,dim=1)
        vals=prob.values.unsqueeze(1)
        indices=prob.indices

        text_inputs  = torch.cat([tokenize(f"{self.prompt_prefix} a photo of a {self.Class_names[c]}") for c in indices])
        image_features = self.network(self.featurizer.encode_image(x))
        # text_embedding = self.featurizer.token_embedding(text_inputs)
    
        fused_feat=self.encode_text_with_image(text_inputs,image_features)
        
        # text_features=vals*text_features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True
        outs=self.classifier(fused_feat)
        return outs


class ERM_clip_weighted_text_conc(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_weighted_text_conc, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.featurizer_orig = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        else:

            self.featurizer.network.head=nn.Identity()
        self.classifier = networks.Classifier(
            1280,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
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
            vis_text_proj = image_features @ self.featurizer_orig.visual.proj

            image_features_ze = vis_text_proj / vis_text_proj.norm(dim=1, keepdim=True)
            text_features_zero_shot = text_features_zero_shot / text_features_zero_shot.norm(dim=1, keepdim=True)
            logit_scale = self.featurizer_orig.logit_scale.exp()
        
            logits_per_image = logit_scale * image_features_ze @ text_features_zero_shot.t()
            prob=F.softmax(logits_per_image,dim=1)
            prob=torch.max(prob,dim=1)
            vals=prob.values.unsqueeze(1)
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

class ERM_clip_weighted_text_confid(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_weighted_text_confid, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.featurizer_orig = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        else:

            self.featurizer.network.head=nn.Identity()
        self.classifier = networks.Classifier(
            1280,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        image_features = self.featurizer.encode_image(all_x)
        
        
        with torch.no_grad():
            text_features = self.featurizer.encode_text(text_inputs)
            text_inputs_zero_sht = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            text_features_zero_shot = self.featurizer.encode_text(text_inputs_zero_sht)
            vis_text_proj = image_features @ self.featurizer_orig.visual.proj

            image_features_ze = vis_text_proj / vis_text_proj.norm(dim=1, keepdim=True)
            text_features_zero_shot = text_features_zero_shot / text_features_zero_shot.norm(dim=1, keepdim=True)
            logit_scale = self.featurizer_orig.logit_scale.exp()
        
            logits_per_image = logit_scale * image_features_ze @ text_features_zero_shot.t()
            prob=F.softmax(logits_per_image,dim=1)
            prob=torch.max(prob,dim=1)
            vals=prob.values.unsqueeze(1)
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

class ERM_clip_weighted_text_label_confid(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_weighted_text_label_confid, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.featurizer_orig = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        else:

            self.featurizer.network.head=nn.Identity()
        self.classifier = networks.Classifier(
            1280,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        image_features = self.featurizer.encode_image(all_x)
        
        
        with torch.no_grad():
            text_features = self.featurizer.encode_text(text_inputs)
            text_inputs_zero_sht = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            text_features_zero_shot = self.featurizer.encode_text(text_inputs_zero_sht)
            vis_text_proj = image_features @ self.featurizer_orig.visual.proj

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

class ERM_clip_patch_tokens_weighted_text_label_confid(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_patch_tokens_weighted_text_label_confid, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.featurizer_orig = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        else:

            self.featurizer.network.head=nn.Identity()
        self.classifier = networks.Classifier(
            2048,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        image_features = self.featurizer.encode_image(all_x,return_all_token=True)
        patch_tok=torch.mean(image_features[:, 1:, :],dim=1)
        image_features=image_features[:, 0, :]
        
        
        text_features = self.featurizer.encode_text(text_inputs)
        
        with torch.no_grad():
            text_inputs_zero_sht = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            text_features_zero_shot = self.featurizer.encode_text(text_inputs_zero_sht)
            vis_text_proj = image_features @ self.featurizer_orig.visual.proj

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
        conc_feat=torch.cat([image_features,patch_tok,text_features],dim=1)
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
        image_features = self.featurizer.encode_image(x,return_all_token=True)
        text_features = self.featurizer.encode_text(text_inputs)
        text_features=vals*text_features
        patch_tok=torch.mean(image_features[:, 1:, :],dim=1)
        image_features=image_features[:, 0, :]
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        conc_feat=torch.cat([image_features,patch_tok,text_features],dim=1)
        outs=self.classifier(conc_feat)
        return outs

class ERM_clip_patch_tokens_weighted_text_label_max_confid(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_patch_tokens_weighted_text_label_max_confid, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network
        self.featurizer_orig = networks.ViT(input_shape, self.hparams,num_classes).network
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None
        else:

            self.featurizer.network.head=nn.Identity()
        self.classifier = networks.Classifier(
            2048,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.parameters())+list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
        image_features = self.featurizer.encode_image(all_x,return_all_token=True)
        patch_tok=torch.mean(image_features[:, 1:, :],dim=1)
        image_features=image_features[:, 0, :]
        
        
        text_features = self.featurizer.encode_text(text_inputs)
        
        with torch.no_grad():
            text_inputs_zero_sht = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to("cuda")
            text_features_zero_shot = self.featurizer.encode_text(text_inputs_zero_sht)
            vis_text_proj = image_features @ self.featurizer_orig.visual.proj

            image_features_ze = vis_text_proj / vis_text_proj.norm(dim=1, keepdim=True)
            text_features_zero_shot = text_features_zero_shot / text_features_zero_shot.norm(dim=1, keepdim=True)
            logit_scale = self.featurizer_orig.logit_scale.exp()
        
            logits_per_image = logit_scale * image_features_ze @ text_features_zero_shot.t()
            prob=F.softmax(logits_per_image,dim=1)
            # vals=prob[torch.arange(len(all_x)), all_y].unsqueeze(1)
  
            prob=torch.max(prob,dim=1)
            vals=prob.values.unsqueeze(1)
            # print(vals.shape)
        text_features=vals*text_features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        conc_feat=torch.cat([image_features,patch_tok,text_features],dim=1)
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
        image_features = self.featurizer.encode_image(x,return_all_token=True)
        text_features = self.featurizer.encode_text(text_inputs)
        text_features=vals*text_features
        patch_tok=torch.mean(image_features[:, 1:, :],dim=1)
        image_features=image_features[:, 0, :]
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)
        conc_feat=torch.cat([image_features,patch_tok,text_features],dim=1)
        outs=self.classifier(conc_feat)
        return outs

class CLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model =  networks.ViT(input_shape, self.hparams,num_classes).network

        # for param in self.clip_model.parameters():
        #     param.requires_grad = False
        
        # print('Set self.clip_model.parameters.reguires_grad = False!')

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512  # 
        self.Class_names=misc.Class_names
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
        self.Class_names=misc.Class_names
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
        image_features = [self.clip_model.encode_image(x) for x in all_x]
        
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
        image_feature = self.clip_model.encode_image(x)
        
        domain_feature = self.network(image_feature)
        mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.Class_names), dim=0)
        text_feature = self._get_text_features(mean_domain_feature)
        
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return self.clip_model.logit_scale.exp() * image_feature @ text_feature.t()

# 
class ERM_clip_WTC_DPL(DPLCLIP):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_WTC_DPL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network

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
        self.Class_names=misc.Class_names
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
        # print(logits_per_image.shape)
        # print(all_y)
        loss = F.cross_entropy(logits_per_image, all_y_d)
        ######
        text_features=torch.index_select(text_features, 0, all_y_d)
        text_features=vals*text_features
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
        text_feature=vals*text_feature

        conc_feat=torch.cat([image_features,text_feature],dim=1)
        outs=self.classifier(conc_feat)
        return outs

class ERM_clip_WTC_DPL_single_net(DPLCLIP):
    """
    Empirical Risk Minimization (ERM)
    """
    

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_clip_WTC_DPL_single_net, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        # self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network

        self.classifier = networks.Classifier(
            1280,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        printNetworkParams(self.clip_model)
        self.optimizer = torch.optim.AdamW(
            list(self.clip_model.parameters())+list(self.classifier.parameters())+list(self.network.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # self.optimizer = torch.optim.SGD(
        #     self.network.parameters(),
        #     lr=self.hparams["lr"],
        #     momentum=self.hparams["momentum"]
        # )
        self.Class_names=misc.Class_names
        # print(self.Class_names)
        self.cnt=0


    def update(self, minibatches, unlabeled=None):
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        # text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
 
        #######
        label_chunks=torch.chunk(all_y,chunks=self.num_domains,dim=0)
        all_y_d =[[len(self.Class_names)*i+y for y in label_chunks[i]] for i in range(self.num_domains)]
        all_y_d=torch.tensor(list(itertools.chain.from_iterable(all_y_d))).cuda().long()
    
        #  encode image for each domain.
        image_features = [self.clip_model.encode_image(x)  for x in all_x]
        vis_text_proj=[x @ self.clip_model.visual.proj for x in image_features]
        image_features=torch.cat(image_features)
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
        text_features=vals*text_features

        conc_feat=torch.cat([image_features,text_features],dim=1)
        loss+=F.cross_entropy(self.classifier(conc_feat), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cnt+=1
        return {'loss': loss.item()}


    def predict(self, x):

        image_feature_conc = self.clip_model.encode_image(x) 
        image_feature=image_feature_conc@ self.clip_model.visual.proj
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
        # image_features = self.featurizer.encode_image(x)

        text_feature=torch.index_select(text_feature, 0, indices)
        text_feature=vals*text_feature

        conc_feat=torch.cat([image_feature_conc,text_feature],dim=1)
        outs=self.classifier(conc_feat)
        return outs

class ERM_clip_WTC_DPL_no_conf(ERM_clip_WTC_DPL_single_net):
    """
    ERM_clip_WTC_DPL_test_conf : This is with single clip model
    """
    

    def update(self, minibatches, unlabeled=None):
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        # text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in all_y]).to("cuda")
 
        #######
        label_chunks=torch.chunk(all_y,chunks=self.num_domains,dim=0)
        all_y_d =[[len(self.Class_names)*i+y for y in label_chunks[i]] for i in range(self.num_domains)]
        all_y_d=torch.tensor(list(itertools.chain.from_iterable(all_y_d))).cuda().long()
    
        #  encode image for each domain.
        image_features = [self.clip_model.encode_image(x)  for x in all_x]
        vis_text_proj=[x @ self.clip_model.visual.proj for x in image_features]
        image_features=torch.cat(image_features)
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

        # prob=F.softmax(logits_per_image,dim=1)
        # vals=prob[torch.arange(len(vis_text_proj)), all_y_d].unsqueeze(1)
     
        loss = F.cross_entropy(logits_per_image, all_y_d)
        ######
        text_features=torch.index_select(text_features, 0, all_y_d)
        # text_features=vals*text_features

        conc_feat=torch.cat([image_features,text_features],dim=1)
        loss+=F.cross_entropy(self.classifier(conc_feat), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cnt+=1
        return {'loss': loss.item()}
    
    def predict(self, x):
    
        image_feature_conc = self.clip_model.encode_image(x) 
        image_feature=image_feature_conc@ self.clip_model.visual.proj
        domain_feature = self.network(image_feature)

        mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.Class_names), dim=0)
        text_feature = self._get_text_features(mean_domain_feature)
        
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)
        logits_per_image=self.clip_model.logit_scale.exp() * image_feature @ text_feature_norm.t()

        prob=F.softmax(logits_per_image,dim=1)
     
        indices=prob.indices
        # text_inputs  = torch.cat([tokenize(f"a photo of a {self.Class_names[c]}") for c in indices]).to("cuda")
        # image_features = self.featurizer.encode_image(x)

        text_feature=torch.index_select(text_feature, 0, indices)
        

        conc_feat=torch.cat([image_feature_conc,text_feature],dim=1)
        outs=self.classifier(conc_feat)
        return outs


class ERM_LowResolution_Pre(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_LowResolution_Pre, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'],init=self.hparams['weight_init'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        # print(self.network)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        if(self.cnt<5000):
            t=transforms.Resize((48,48))
            all_x=t(all_x)
        elif(self.cnt<10000):
            t=transforms.Resize((112,112))
            all_x=t(all_x)
        elif(self.cnt<15000):
            t=transforms.Resize((160,160))
            all_x=t(all_x)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class ERM_ViT_LowResolution_Pre(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_ViT_LowResolution_Pre, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.network = networks.ViT(input_shape, self.hparams,num_classes)
        printNetworkParams(self.network)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.cnt=0
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        if(self.cnt<5000):
            t=transforms.Resize((48,48))
            all_x=t(all_x)
        elif(self.cnt<10000):
            t=transforms.Resize((112,112))
            all_x=t(all_x)
        elif(self.cnt<15000):
            t=transforms.Resize((160,160))
            all_x=t(all_x)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain 
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}

class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):
        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)
            
            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)

            env_grads = autograd.grad(env_loss, self.network.parameters())
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {'loss': mean_loss}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=False):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(env_loss, self.network.parameters(), 
                                        create_graph=True)

            grads.append(env_grad)
            
        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(mean_loss, self.network.parameters(), 
                                        retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}
    
    
class SelfReg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = input_feat_size if input_feat_size==2048 else input_feat_size*2
        
        self.cdpl = nn.Sequential(
                            nn.Linear(input_feat_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_size, input_feat_size),
                            nn.BatchNorm1d(input_feat_size)
        )
        
    def update(self, minibatches, unlabeled=None):
        
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)
        
        batch_size = all_y.size()[0]
        
        # cluster and order features into same-class group
        with torch.no_grad():   
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex==val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y
        
        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)
        
        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end-ex)+ex
            shuffle_indices2 = torch.randperm(end-ex)+ex
            for idx in range(end-ex):
                output_2[idx+ex] = output[shuffle_indices[idx]]
                feat_2[idx+ex] = proj[shuffle_indices[idx]]
                output_3[idx+ex] = output[shuffle_indices2[idx]]
                feat_3[idx+ex] = proj[shuffle_indices2[idx]]
            ex = end
        
        # mixup 
        output_3 = lam*output_2 + (1-lam)*output_3
        feat_3 = lam*feat_2 + (1-lam)*feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)
        
        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale*(lam*(L_ind_logit + L_ind_feat)+(1-lam)*(L_hdl_logit + L_hdl_feat))
     
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            betas=betas
        )

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):

        mean_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        self.optimizer.zero_grad()
        # gradient masking applied here
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {'loss': mean_loss}

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=False):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)

class TRM(Algorithm):
    """
    Learning Representations that Support Robust Transfer of Predictors
    <https://arxiv.org/abs/2110.09940>
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.num_domains = num_domains
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer.n_outputs, num_classes).cuda() for i in range(num_domains+1)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(num_domains+1)]

        self.optimizer_f = torch.optim.Adam(
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()

    @staticmethod
    def neum(v, model, batch):
        def hvp(y, w, v):

            # First backprop
            first_grads = autograd.grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
            first_grads = torch.nn.utils.parameters_to_vector(first_grads)
            # Elementwise products
            elemwise_products = first_grads @ v
            # Second backprop
            return_grads = autograd.grad(elemwise_products, w, create_graph=True)
            return_grads = torch.nn.utils.parameters_to_vector(return_grads)
            return return_grads

        v = v.detach()
        h_estimate = v
        cnt = 0.
        model.eval()
        iter = 10
        for i in range(iter):
            model.weight.grad *= 0
            y = model(batch[0].detach())
            loss = F.cross_entropy(y, batch[1].detach())
            hv = hvp(loss, model.weight, v)
            v -= hv
            v = v.detach()
            h_estimate = v + h_estimate
            h_estimate = h_estimate.detach()
            # not converge
            if torch.max(abs(h_estimate)) > 10:
                break
            cnt += 1

        model.train()
        return h_estimate.detach()

    def update(self, minibatches):

        loss_swap = 0.0
        trm = 0.0

        if self.update_count >= self.hparams['iters']:
            # TRM
            if self.hparams['class_balanced']:
                # for stability when facing unbalanced labels across environments
                for classifier in self.clist:
                    classifier.weight.data = copy.deepcopy(self.classifier.weight.data)
            self.alpha /= self.alpha.sum(1, keepdim=True)
            
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            # updating original network
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

            for i in range(30):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature.detach()), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            # collect (feature, y)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            # estimate transfer risk
            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)

                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach())
                          if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P).detach()
                vec_grad_P = self.neum(vec_grad_P, self.clist[Q], (feature_split[Q], y_split[Q]))

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q)

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()

            loss_swap /= len(minibatches)
            trm /= len(minibatches)
        else:
            # ERM
            self.featurizer.train()
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_feature = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(all_feature), all_y)

        nll = loss.item()
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap)
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        loss_swap = loss_swap.item() - nll
        self.update_count += 1

        return {'nll': nll, 'trm_loss': loss_swap}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

    def train(self):
        self.featurizer.train()

    def eval(self):
        self.featurizer.eval()

class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll 
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 
                'nll': nll.item(),
                'IB_penalty': ib_penalty.item()}

class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll 
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 
                'nll': nll.item(),
                'IRM_penalty': irm_penalty.item(), 
                'IB_penalty': ib_penalty.item()}

def printNetworkParams(net):
    pytorch_total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("pytorch_total_trainable_params:",pytorch_total_trainable_params)


def load_model(fname):
    dump = torch.load(fname)
    algorithm_class = get_algorithm_class(dump["args"]["algorithm"])
    algorithm = algorithm_class(
        dump["model_input_shape"],
        dump["model_num_classes"],
        dump["model_num_domains"],
        dump["model_hparams"])
    algorithm.load_state_dict(dump["model_dict"],strict=False)
    return algorithm