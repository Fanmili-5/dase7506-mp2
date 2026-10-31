"""Real-data alternatives to correct-set mass, with unchanged neural capacity."""
import torch
from torch import nn
from torch.nn import functional as F
from joint_verifier_network import Network


def make_network(objective):
    model=Network('joint')
    if objective=='answer_group':model.register_parameter('vote_logit',nn.Parameter(torch.zeros(())))
    return model


def pooled(objective, model, scores, same, valid):
    if objective=='pairwise':return scores.mean(dim=1)
    group_max=scores[:,:,None,:].masked_fill(~same[:,None].bool(),-1e4).amax(dim=-1)
    counts=same.sum(-1).clamp(min=1)
    return (group_max+model.vote_logit.sigmoid()*counts[:,None].log()).mean(dim=1)


def objective_for(objective, model, scores, labels, valid, same):
    if objective=='pairwise':
        mask=labels.bool()[:,:,None] & (~labels.bool())[:,None,:] & valid[:,:,None] & valid[:,None,:]
        differences=scores[:,:,:,None]-scores[:,:,None,:]
        main=((F.softplus(-differences)*mask[:,None]).sum(dim=(-1,-2))/mask.sum(dim=(-1,-2))[:,None].clamp(min=1)).mean()
    else:
        first=valid & ~(same.bool() & torch.tril(torch.ones(4,4,dtype=torch.bool,device=scores.device),diagonal=-1)).any(dim=-1)
        group_max=scores[:,:,None,:].masked_fill(~same[:,None].bool(),-1e4).amax(dim=-1)
        group_score=group_max+model.vote_logit.sigmoid()*same.sum(-1).clamp(min=1)[:,None].log()
        group_score=group_score.masked_fill(~first[:,None],-1e4)
        target=(labels.bool() & first).to(torch.int64).argmax(dim=-1)
        main=F.cross_entropy(group_score.reshape(-1,4),target[:,None].expand(-1,scores.shape[1]).reshape(-1))
    bce=F.binary_cross_entropy_with_logits(scores,labels[:,None].expand_as(scores),reduction='none')
    bce=(bce*valid[:,None]).sum()/valid.sum()/scores.shape[1]
    return main+.1*bce
