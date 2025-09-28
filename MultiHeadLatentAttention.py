import torch
import torch.nn as nn
class multiheadlatentattention(nn.Module):
    def __init__(self,d_in,d_out,dl,num_head,dropout_rate,contextlenght):
        self.num_head=num_head
        self.dl=dl
        self.h_d_vkq=self.dl//num_head
        self.dropout=nn.Dropout(dropout_rate)
        self.w_key=nn.Linear(d_in,d_out)
        self.w_kl=nn.Linear(d_out,self.dl)
        self.w_query=nn.Linear(d_in,d_out)
        self.w_value=nn.Linear(d_in,d_out)
        self.w_vl=nn.Linear(d_out,self.dl)
        self.w_ql=nn.Linear(d_out,self.dl)
        self.register_buffer("mask",torch.tril(torch.ones(contextlenght,contextlenght)))
        self.w_out=nn.Linear(self.dl,d_out)
    def forward(self,x):
        b,num_token,d_in=x.shape
        key=self.w_key(x)@self.w_kl
        query=self.w_query(x)@self.w_ql
        value=self.w_value(x)@self.w_vl
        key=key.view(b,num_token,self.num_head,self.h_d_vkq)
        query=query.view(b,num_token,self.num_head,self.h_d_vkq)
        value=value.view(b,num_token,self.num_head,self.h_d_vkq)
        key=key.transpose(1,2)
        query=query.transpose(1,2)
        value=value.transpose(1,2)
        at=query@key.transpose(2,3)
        at.masked_fill_(self.mask.bool()[:num_token,:num_token]==0,-torch.inf)
        at=torch.softmax(at/key.shape[-1]**0.5,dim=-1)
        at=self.dropout(at)
        at_score=((at@value).transpose(1,2)).contiguous()
        at_score=at_score@self.w_out
        at_score=at_score.view(b,num_token,self.d_out)
        return at_score