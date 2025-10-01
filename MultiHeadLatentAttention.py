import torch
import torch.nn as nn
class multiheadlatentattention(nn.Module):
    def __init__(self,d_in,d_out,dl,num_head,dropout_rate,contextlenght):
        super().__init__()
        self.d_out=d_out
        self.num_head=num_head
        self.dl=dl
        self.head_dim=d_out//num_head
        self.dropout=nn.Dropout(dropout_rate)
        self.w_kv=nn.Linear(d_in,dl)
        self.w_ku=nn.Linear(dl,d_out//2)
        self.w_kr=nn.Linear(d_in,d_out//2)
        self.w_vu=nn.Linear(dl,d_out)
        self.w_query=nn.Linear(d_in,dl)
        self.w_qu=nn.Linear(dl,d_out//2)
        self.w_qr=nn.Linear(dl,d_out//2)
        self.rope=Rope(contextlenght,d_out//2)
        self.register_buffer("mask",torch.tril(torch.ones(contextlenght,contextlenght)))
        self.w_out=nn.Linear(d_out,d_out)
    def forward(self,x):
        b,num_token,d_in=x.shape
        ckv=self.w_kv(x)
        key=self.w_ku(ckv)
        key_r=self.rope(self.w_kr(x))
        key=torch.concat([key,key_r],dim=-1)
        query=self.w_query(x)
        query_u=self.w_qu(query)
        query_r=self.rope(self.w_qr(query))
        query=torch.concat([query_u,query_r],dim=-1)
        value=self.w_vu(ckv)
        key=key.view(b,num_token,self.num_head,self.head_dim)
        query=query.view(b,num_token,self.num_head,self.head_dim)
        value=value.view(b,num_token,self.num_head,self.head_dim)
        key=key.transpose(1,2)
        query=query.transpose(1,2)
        value=value.transpose(1,2)
        at=query@key.transpose(2,3)
        at.masked_fill_(self.mask.bool()[:num_token,:num_token]==0,-torch.inf)
        at=torch.softmax(at/key.shape[-1]**0.5,dim=-1)
        at=self.dropout(at)
        at_score=((at@value).transpose(1,2)).contiguous()
        at_score=self.w_out(at_score)
        at_score=at_score.view(b,num_token,self.d_out)
        return at_score
