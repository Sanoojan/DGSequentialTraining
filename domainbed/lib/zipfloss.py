import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import megengine 
# import megengine.functional as F

# from megengine import Tensor

def zipf_loss(targetlogits,logits,  labels,feats=None, loss_mask=False, dense=True):
    non_target_pdf = gen_pdf(targetlogits, feats, labels, dense=dense)
    dist_loss = kl_loss(logits, labels, non_target_pdf, do_mask=loss_mask)
    return dist_loss

def kl_div(p, q):
    return q * (torch.log(q)-F.log_softmax(p, dim=1))

def kl_loss(output, label, pdf, do_mask=False):
    mask = torch.ones_like(output)
    tmp = torch.unsqueeze(label,dim=1)
    mask=mask.scatter_( 1, tmp, torch.zeros_like(tmp).float())
    res = output[mask.type(torch.bool)].reshape((output.shape[0], output.shape[1]-1))
    res = F.log_softmax(res, dim=1)
    loss = kl_div(res, pdf).sum(dim=1)

    if do_mask:
        loss_mask = (torch.argmax(output, dim=1) == label)
        if loss_mask.any():
            loss = loss * loss_mask
            loss = loss.sum() / loss_mask.sum()
        else:
            loss = loss.mean() * 0
    else:
        loss = loss.mean()

    return loss

def gen_pdf(logits, feats, labels, dense=True):
    n = logits.shape[0]
    n, c = logits.shape
    
    mask = torch.ones_like(logits)
    tmp = torch.unsqueeze(labels, dim=1)
    mask=mask.scatter_( 1, tmp, torch.zeros_like(tmp).float())
    
    if dense:
        d_n, d_count = feats.shape
        assert n == d_n
        feats = F.expand_dims(feats, axis=2).int()
        logits = F.zeros((n,d_count,c)).int()
        F.scatter(logits, 2, feats, F.ones_like(feats).int())
        logits = logits.sum(1)[:,:c]
   
    rank = rank_data(-logits)
    # print(rank)
    # print(mask)
    # zipf dist
    power = 1.0
    dist = (1 / rank) ** power
    dist = dist[mask.type(torch.bool)].reshape((n, c-1))
    n, c = rank.shape
    dist = dist / dist.sum(dim=1, keepdims=True)
    return dist

def rank_data(x):
    # print(x.shape)
    device = x.device
    vals,inds=torch.sort(x)    
    rank=torch.zeros(x.shape,device=device).int()
    temp_inds = torch.arange(end=x.shape[1]+1,dtype=torch.int32,device=device)
    # print(temp_inds.shape)
    temp_inds = torch.unsqueeze(temp_inds, dim=0)
    temp_inds = torch.broadcast_to(temp_inds, [x.shape[0],x.shape[1]+1])
    rank=rank.scatter_(1,inds,temp_inds[:,:-1].int())
    # print(rank)
    obs=(vals[:,1:]!=vals[:,:-1])
    # print(obs)
    all_true = torch.ones([x.shape[0],1],dtype=bool,device=device)
    obs=torch.cat([all_true,obs],dim=1).int()
    
    obs_cum=torch.cumsum(obs,axis=1)
    obs_cum_mask=obs_cum*obs
    temp3=torch.zeros(temp_inds.shape,dtype=torch.int32,device=device)
    temp_inds, obs_cum_mask
    temp3=temp3.scatter_( 1,obs_cum_mask,temp_inds[:,:-1])
    dense=torch.gather(torch.cumsum(obs, axis=1).type(torch.int64),1,rank.type(torch.int64))
    rank_data=torch.gather(temp3.type(torch.int64),1,dense) + 1
    return rank_data

