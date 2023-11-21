# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest.mock import patch
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import copy
import numpy as np
import einops
import os
import itertools
import domainbed.lib.clip.clip as clip
from collections import defaultdict, OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None
import wandb
from concurrent.futures import ThreadPoolExecutor, wait

from domainbed.lib.visiontransformer import *
from domainbed import networks
from domainbed.lib.clip.clip import tokenize
from domainbed import global_var
from domainbed.lib import misc
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)
try:
    # from transformers import AutoImageProcessor, BeitForImageClassification
    from transformers import OFAModel
    # from transformers.models.ofa.generate import sequence_generator
except:
    print("No OFA Model ==============================================")
    AutoImageProcessor=None
    BeitForImageClassification=None


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# from DOSNES import dosnes




ALGORITHMS = [
    'ERM',
    'Clip_train_mixup_with_text',
    'Clip_train'
    
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
        # self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network.visual
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/16', device)
        self.featurizer=model.float()

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


class Clip_train(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """


    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network.to(device_0)
        if(self.hparams['weight_init']=="clip_full"):
            print("clip_full")
            # self.featurizer.network.proj=None

        printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.featurizer.visual.parameters()),
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
        
        time_start=time.time()
       
        image_features = self.featurizer.encode_image(all_x)
      
        image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        wandb.log({"forward_time":time.time()-time_start})
        time_start=time.time()
        
        self.optimizer.zero_grad()
        loss.backward()
        wandb.log({"backward_time":time.time()-time_start})
        self.optimizer.step()
        time_start=time.time()
        wandb.log({"update_time":time.time()-time_start})
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

class Clip_train_No_distributed(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """


    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train_No_distributed, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        device_0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # breakpoint()
        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network.to(device_0)
        self.featurizer.visual.to(device_0)
        
        # self.text_featurizer=self.featurizer.text.to(device_1)
        # breakpoint()
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
        self.text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to(device_0)
        
        # breakpoint()
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        
        # with ThreadPoolExecutor() as executor:
        #     futures = []
        #     futures.append(executor.submit(train, dense, dummy1, 1000))
        #     futures.append(executor.submit(train, rest, dummy2, 1000))
        #     complete_futures, incomplete_futures = wait(futures)
        #     for f in complete_futures:
        #         output.append(f.result())
        #         print(str(f.result()))

        # elapsed = (time.time() - start_time)
        # print(f"Total time of execution {round(elapsed, 4)} second(s)")
        # print("Output is:",output)

        # print(self.cnt)
        
        time_start=time.time()
        
        text_features = self.featurizer.encode_text(self.text_inputs,cnt=self.cnt)
        image_features = self.featurizer.encode_image(all_x,cnt=self.cnt)
      
        image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()


        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        wandb.log({"forward_time":time.time()-time_start})
        # print("time_forward:cnt",self.cnt,":",time.time()-time_start)
        time_start=time.time()
        self.optimizer.zero_grad()
        loss.backward()
        wandb.log({"backward_time":time.time()-time_start})
        time_start=time.time()
        # print("time_backward:cnt",self.cnt,":",time.time()-time_start)
        self.optimizer.step()
        wandb.log({"update_time":time.time()-time_start})
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
 
        
        image_features = self.featurizer.encode_image(x)
        
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj
        text_features = self.featurizer.encode_text(self.text_inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()
        logit_scale = logit_scale.to("cuda:0")
        text_features=text_features.to("cuda:0")
        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image



class Clip_train_distributed(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """


    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train_distributed, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # breakpoint()
        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network.to(device_1)
        self.featurizer.visual.to(device_0)
        
        # self.text_featurizer=self.featurizer.text.to(device_1)
        # breakpoint()
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
        self.text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to(device_1)
        
        # breakpoint()
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        
        # with ThreadPoolExecutor() as executor:
        #     futures = []
        #     futures.append(executor.submit(train, dense, dummy1, 1000))
        #     futures.append(executor.submit(train, rest, dummy2, 1000))
        #     complete_futures, incomplete_futures = wait(futures)
        #     for f in complete_futures:
        #         output.append(f.result())
        #         print(str(f.result()))

        # elapsed = (time.time() - start_time)
        # print(f"Total time of execution {round(elapsed, 4)} second(s)")
        # print("Output is:",output)

        # print(self.cnt)
        
        time_start=time.time()
        
        text_features = self.featurizer.encode_text(self.text_inputs,cnt=self.cnt)
        image_features = self.featurizer.encode_image(all_x,cnt=self.cnt)
      
        image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()
        logit_scale = logit_scale.to("cuda:0")
        text_features=text_features.to("cuda:0")
        # breakpoint()
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        wandb.log({"forward_time":time.time()-time_start})
        # print("time_forward:cnt",self.cnt,":",time.time()-time_start)
        time_start=time.time()
        self.optimizer.zero_grad()
        loss.backward()
        wandb.log({"backward_time":time.time()-time_start})
        time_start=time.time()
        # print("time_backward:cnt",self.cnt,":",time.time()-time_start)
        self.optimizer.step()
        wandb.log({"update_time":time.time()-time_start})
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
 
        
        image_features = self.featurizer.encode_image(x)
        
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj
        text_features = self.featurizer.encode_text(self.text_inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()
        logit_scale = logit_scale.to("cuda:0")
        text_features=text_features.to("cuda:0")
        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image
    
    


class Clip_train_distributed_async(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """


    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Clip_train_distributed_async, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # breakpoint()
        self.featurizer = networks.ViT(input_shape, self.hparams,num_classes).network.to(device_1)
        self.featurizer.visual.to(device_0)
        
        # self.text_featurizer=self.featurizer.text.to(device_1)
        # breakpoint()
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
        self.text_inputs  = torch.cat([tokenize(f"a photo of a {c}") for c in self.Class_names]).to(device_1)
        
        # breakpoint()
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
        
        def exec(model,x,cnt):
            ret=model(x,cnt=cnt)
            return ret
        time_start=time.time()
        with ThreadPoolExecutor() as executor:
            futures = []
            futures.append(executor.submit(exec, self.featurizer.encode_text, self.text_inputs,self.cnt))
            futures.append(executor.submit(exec, self.featurizer.encode_image, all_x,self.cnt))
            done, _ = wait(futures)
            for future in done:
                result = future.result()
                if future in futures:
                    index = futures.index(future)
                    if index == 0:
                        text_features = result
                    elif index == 1:
                        image_features = result

      
        image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.featurizer.logit_scale.exp()
        logit_scale = logit_scale.to("cuda:0")
        text_features=text_features.to("cuda:0")
        # breakpoint()
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        wandb.log({"forward_time":time.time()-time_start})
        time_start=time.time()
        
        self.optimizer.zero_grad()
        loss.backward()
        wandb.log({"backward_time":time.time()-time_start})
        time_start=time.time()
        self.optimizer.step()
        wandb.log({"update_time":time.time()-time_start})
        self.cnt+=1
        return {'loss': loss.item()}

    def predict(self, x):
 
        
        image_features = self.featurizer.encode_image(x)
        
        # text_features = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)] @ self.network.text_projection
        image_features = image_features @ self.featurizer.visual.proj
        text_features = self.featurizer.encode_text(self.text_inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

        # cosine similarity as logits
        logit_scale = self.featurizer.logit_scale.exp()
        logit_scale = logit_scale.to("cuda:0")
        text_features=text_features.to("cuda:0")
        logits_per_image = logit_scale * image_features @ text_features.t()
        
        return logits_per_image




# Helpers ......................................................
def printNetworkParams(net):
    pytorch_total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("pytorch_total_trainable_params:",pytorch_total_trainable_params)

class Testing(Algorithm): 


    def __init__(self, input_shape, num_classes, num_domains, hparams, input_dir=None):
        super(Testing, self).__init__(input_shape, num_classes, num_domains,
                                           hparams)

        lists=os.listdir(input_dir)
        model_name='IID_best.pkl'
        for mod_name in lists:
            if "best" in mod_name:
                model_name=mod_name
                break

        

        algo=load_model(input_dir+model_name)
        self.network=algo.featurizer
 
        self.featurizer=copy.deepcopy(self.network)
        # self.featurizer.head=Identity()
        self.text_features=algo.text_features

        self.classifier=torch.nn.Linear(768,512)
        self.classifier.weight = self.network.visual.proj
        
        # self.network.head = nn.Linear(384, num_classes)
        # self.network.head_dist = nn.Linear(384, num_classes)  # reinitialize the last layer

        self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                           lr=self.hparams["lr"],
                                           weight_decay=self.hparams['weight_decay']
                                           )

    def update(self, minibatches, unlabeled=None):
        return None

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

class Identity(torch.nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


