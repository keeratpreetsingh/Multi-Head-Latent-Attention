import torch
import torch.nn as nn
class multiheadlatentattention(nn.Module):
    def __init__(self,d_in,d_out,dl,num_head,dropout_rate,contextlenght):
        super().__init__()
        self.d_out=d_out
        self.num_head=num_head
        self.dl=dl
        self.d_head=d_out//num_head
        self.dropout=nn.Dropout(dropout_rate)
        self.w_key=nn.Linear(d_in,d_out)
        self.w_kl=nn.Linear(d_out,self.dl)
        self.w_query=nn.Linear(d_in,d_out)
        self.w_value=nn.Linear(d_in,d_out)
        self.w_vl=nn.Linear(d_out,self.dl)
        self.w_up_k = nn.ModuleList([nn.Linear(dl, self.d_head) for i in range(num_head)])
        self.w_up_v = nn.ModuleList([nn.Linear(dl, self.d_head) for i in range(num_head)])
        self.register_buffer("mask",torch.tril(torch.ones(contextlenght,contextlenght)))
        self.w_out=nn.Linear(d_out,d_out)
    def forward(self,x):
        b,num_token,d_in=x.shape
        key=self.w_kl(self.w_key(x))
        query=self.w_query(x)
        value=self.w_vl(self.w_value(x))
        k_heads = [self.w_up_k[h](key).view(b, num_token, 1, self.d_head) for h in range(self.num_head)]
        v_heads = [self.w_up_v[h](value).view(b, num_token, 1, self.d_head) for h in range(self.num_head)]
        key = torch.cat(k_heads, dim=2)  
        value = torch.cat(v_heads, dim=2)
        query=query.view(b,num_token,self.num_head,self.d_head)
        
        key=key.transpose(1,2)
        query=query.transpose(1,2)
        value=value.transpose(1,2)
        at=query@key.transpose(2,3)
        at.masked_fill_(self.mask.bool()[:num_token,:num_token]==0,-torch.inf)
        at=torch.softmax(at/key.shape[-1]**0.5,dim=-1)
        at=self.dropout(at)
        at_score=((at@value).transpose(1,2)).contiguous()
        at_score=at_score.view(b,num_token,self.d_out)
        at_score=self.w_out(at_score)
        return at_score
