"""All-new trainable candidate verifier. No frozen parent or teacher logits."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Network(nn.Module):
    def __init__(self, mode, width=128, members=8, dropout=.25):
        super().__init__()
        self.mode,self.dropout=mode,dropout
        self.aw=nn.Parameter(torch.empty(1024,width));self.rw=nn.Parameter(torch.empty(1024,width))
        self.sw=nn.Parameter(torch.empty(32,width))
        self.ar=nn.Parameter(torch.empty(members,1024));self.rr=nn.Parameter(torch.empty(members,1024))
        self.register_parameter('as',nn.Parameter(torch.ones(members,width)))
        self.rs=nn.Parameter(torch.ones(members,width));self.ib=nn.Parameter(torch.zeros(members,width))
        self.out=nn.Parameter(torch.empty(members,width));self.out_bias=nn.Parameter(torch.zeros(members))
        if mode=='joint':
            self.qkv_w=nn.Parameter(torch.empty(width,3*width));self.qkv_b=nn.Parameter(torch.zeros(3*width))
            self.attn_w=nn.Parameter(torch.empty(width,width));self.attn_b=nn.Parameter(torch.zeros(width))
            self.ff_w=nn.Parameter(torch.empty(width,2*width));self.ff_b=nn.Parameter(torch.zeros(2*width))
            self.fo_w=nn.Parameter(torch.empty(2*width,width));self.fo_b=nn.Parameter(torch.zeros(width))
            self.same_bias=nn.Parameter(torch.zeros(members))
            self.n1_g=nn.Parameter(torch.ones(width));self.n1_b=nn.Parameter(torch.zeros(width))
            self.n2_g=nn.Parameter(torch.ones(width));self.n2_b=nn.Parameter(torch.zeros(width))
        else:
            self.hr=nn.Parameter(torch.ones(members,width));self.hw=nn.Parameter(torch.empty(width,width))
            self.hs=nn.Parameter(torch.ones(members,width));self.hb=nn.Parameter(torch.zeros(members,width))
        with torch.no_grad():
            self.ar.bernoulli_(.5).mul_(2).sub_(1);self.rr.bernoulli_(.5).mul_(2).sub_(1)
            for name,p in self.named_parameters():
                if name.endswith('_w') or name in ('aw','rw','sw','hw'):
                    p.uniform_(-1/math.sqrt(p.shape[0]),1/math.sqrt(p.shape[0]))
            self.out.uniform_(-1/math.sqrt(width),1/math.sqrt(width))

    def forward(self, absolute, relative, scalar, same, valid):
        a,r,s=(x[:,None] for x in (absolute,relative,scalar))
        h=((a*self.ar[None,:,None])@self.aw)*getattr(self,'as')[None,:,None]
        h=h+((r*self.rr[None,:,None])@self.rw)*self.rs[None,:,None]
        h=h+s@self.sw+self.ib[None,:,None]
        h=F.dropout(F.relu(h),self.dropout,self.training)
        if self.mode=='joint':
            q,k,v=(h@self.qkv_w+self.qkv_b).chunk(3,dim=-1)
            attention=(q@k.transpose(-1,-2))/math.sqrt(h.shape[-1])
            attention=attention+self.same_bias[None,:,None,None]*same[:,None]
            attention=attention.masked_fill(~valid[:,None,None,:],-1e4).softmax(-1)
            h=F.layer_norm(h+(attention@v)@self.attn_w+self.attn_b,(h.shape[-1],),self.n1_g,self.n1_b,1e-5)
            hidden=F.dropout(F.relu(h@self.ff_w+self.ff_b),self.dropout,self.training)
            h=F.layer_norm(h+hidden@self.fo_w+self.fo_b,(h.shape[-1],),self.n2_g,self.n2_b,1e-5)
        else:
            h=F.relu(((h*self.hr[None,:,None])@self.hw)*self.hs[None,:,None]+self.hb[None,:,None])
        h=F.dropout(h,self.dropout,self.training)
        return (h*self.out[None,:,None]).sum(-1)+self.out_bias[None,:,None]


def objective(scores, labels, valid):
    pos=labels[:,None].bool() & valid[:,None]
    mass=(scores.masked_fill(~valid[:,None],-1e4).logsumexp(-1)-scores.masked_fill(~pos,-1e4).logsumexp(-1)).mean()
    bce=F.binary_cross_entropy_with_logits(scores,labels[:,None].expand_as(scores),reduction='none')
    bce=(bce*valid[:,None]).sum()/valid.sum()/scores.shape[1]
    return mass+.1*bce
