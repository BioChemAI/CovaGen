import torch
import os,sys
import json
sys.path.append(os.path.dirname(sys.path[0]))
from torch import nn
from optdiffusion.EGNNEncoder_pyg import EGNNEncoder
from optdiffusion.MultiHeadAttentionLayer import MultiHeadAttentionLayer
import numpy as np
from torch import utils
from torch_geometric.utils import to_dense_batch
from dgg.models.encoders.schnet import SchNetEncoder
import torch as th
import math
import esm
from datetime import datetime, date

class Dynamics_revive_samp(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
                 sampling=False):
        super().__init__()
        self.mlp = nn.Linear(target_dim + 1 if condition_time else target_dim, 128)

        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
        self.condition_time = condition_time
        self.out = nn.Sequential(nn.Linear(256, target_dim),
                                 nn.ReLU(),
                                 nn.Linear(128, 128))
        self.conhidadj = nn.Linear(28, 128)

    def set_value(self, cond,ma):
        self.cond = cond
        self.ma= ma

    def forward(self, data, condition_x,condition_pos, noidata, batch, t=None, samp = True, look=1,num_samples=None,fresh_noise=None):
        batch = batch
        noitarget = noidata
        print("now look",look)
        bs = max(batch) + 1
        print("bs:",bs)
        bs = bs*num_samples
        if self.condition_time:
           if np.prod(t.size()) == 1:
               h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
               print("h_time shape:",h_time.shape)
           target_with_time = torch.cat([noitarget, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)
        if look == 0:
            condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
            condition_hidden = self.conhidadj(condition_hidden)
            condition_dense, mask = to_dense_batch(condition_hidden, batch)

            print("orishape of mask",mask.shape)
            condition_dense=condition_dense.repeat(num_samples,1,1)
            mask=mask.repeat(num_samples,1)
            mask = mask.unsqueeze(1).unsqueeze(2)
            self.set_value(condition_dense,mask)


        target_cond, attention = self.attention_model(target_hidden, self.cond, self.cond, self.ma)
        target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
        output = self.out(target_merged)
        error = noidata - output
        return error

class Dynamics(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time
        self.out = nn.Linear(hid_dim, target_dim)
        self.conhidadj = nn.Linear(28, 64)
    def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):
        target = data.target
        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        if self.condition_time:
           if np.prod(t.size()) == 1:
               h_time = torch.empty_like(noidata[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
           target_with_time = torch.cat([noidata, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noidata)

        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        noidata = noidata.unsqueeze(1)
        error = noidata - output
        print("error's shape:",error.shape)
        error = error.squeeze(1)
        print("squeezed shape:",error.shape)

        return error

class Dynamics_samp(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()

        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time
        self.out = nn.Linear(hid_dim, target_dim)
        self.conhidadj = nn.Linear(28, 64)
        
    def forward(self, data, condition_x,condition_pos, noidata, batch, t=None, samp = True, look=1,num_samples=None,fresh_noise=None):
        condition_x = condition_x
        condition_pos = condition_pos

        batch = batch
        target = data.target
        noitarget = noidata

        num_nodes = data.nodes
        bs = max(batch) + 1
        bs = bs*num_samples
        if self.condition_time:
           if np.prod(t.size()) == 1:
               h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
           target_with_time = torch.cat([noitarget, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)
        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        condition_dense=condition_dense.repeat(num_samples,1,1)
        mask=mask.repeat(num_samples,1)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        output1 = output.squeeze(1)
        error = noidata - output1
        if samp:
            return error
        elif not samp:
            return bs

class Dynamics_samp2(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time
        self.out = nn.Linear(hid_dim, target_dim)
        self.conhidadj = nn.Linear(28, 64)
        
    def forward(self, data, condition_x,condition_pos, noidata, batch, t=None, samp = True, look=1,num_samples=None,fresh_noise=None):
        condition_x = condition_x
        condition_pos = condition_pos

        batch = batch
        target = data.target
        noitarget = noidata


        print("now look",look)
        num_nodes = data.nodes
        bs = max(batch) + 1
        print("bs:",bs)
        bs = bs*num_samples
        if self.condition_time:
           if np.prod(t.size()) == 1:
               h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
               print("h_time shape:",h_time.shape)
           target_with_time = torch.cat([noitarget, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)

        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        print("orishape of mask",mask.shape)
        condition_dense=condition_dense.repeat(num_samples,1,1)
        mask=mask.repeat(num_samples,1)
        print("shape de tar_hidden",target_hidden.shape)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        output1 = output.squeeze(1)
        error = output1-noidata
        if samp:
            print("nowwesample")
            return error
        elif not samp:
            return bs

class Dynamics_nocond(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
                 sampling=False):
        super().__init__()
        self.condition_time = condition_time
        self.mlp = nn.Linear(128*65 + 1 if condition_time else target_dim, 1024)
        self.mlp2 = nn.Sequential(nn.Linear(1024, 512),
                                  nn.SiLU(),
                                  nn.Linear(512, 1024),
                                  )
        self.SiLU = nn.SiLU()
        self.out = nn.Linear(1024, 128*65)

    def forward(self, x,t, batch=None):
        bs = batch
        noitarget = x
        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, 1)
                h_time = h_time.view(bs, 1)
                print("h_time shape:", h_time.shape)
            target_with_time = torch.cat([noitarget, h_time], dim=1)
            target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)
        target_hidden = self.SiLU(target_hidden)
        target_hidden = self.mlp2(target_hidden)
        print("shape de tar_hidden", target_hidden.shape)
        output = self.out(target_hidden)
        output1 = output.squeeze(1)
        error = noitarget - output1
        return error

class Dynamics_nocond_samp(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
                 sampling=False):
        super().__init__()
        self.condition_time = condition_time
        self.mlp = nn.Linear(target_dim + 1 if condition_time else target_dim, 128)
        self.mlp2 = nn.Linear(128,128)
        self.SiLU = nn.SiLU()
        self.out = nn.Linear(128, target_dim)

    def forward(self,noidata, t=None, samp=True, look=1,num_samples=None, fresh_noise=None):
        noitarget = noidata
        bs = 10000
        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, 1)
                h_time = h_time.view(bs, 1)
            target_with_time = torch.cat([noitarget, h_time], dim=1)
            target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)
        target_hidden = self.SiLU(target_hidden)
        target_hidden = self.mlp2(target_hidden)
        output = self.out(target_hidden)
        output1 = output.squeeze(1)
        error = noidata - output1
        return error



class Dynamics_noi_samp(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
                 sampling=False):
        super().__init__()
        self.mlp = nn.Linear(target_dim + 1 if condition_time else target_dim, hid_dim)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time
        self.out = nn.Linear(hid_dim, target_dim)
        self.conhidadj = nn.Linear(28, 64)

    def forward(self, data, condition_x, condition_pos, noidata, batch, t=None, samp=True, look=1, num_samples=None,
                fresh_noise=None):
        condition_x = condition_x
        condition_pos = condition_pos

        batch = batch
        target = data.target
        noitarget = noidata

        print("now look", look)
        num_nodes = data.nodes
        bs = max(batch) + 1
        print("bs:", bs)
        bs = bs * num_samples
        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, 1)
                h_time = h_time.view(bs, 1)
                print("h_time shape:", h_time.shape)
            target_with_time = torch.cat([noitarget, h_time], dim=1)
            target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)
        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        print("orishape of mask", mask.shape)
        condition_dense = condition_dense.repeat(num_samples, 1, 1)
        mask = mask.repeat(num_samples, 1)
        print("shape de tar_hidden", target_hidden.shape)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        output1 = output.squeeze(1)
        return output1

def TimestepEmbedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Dynamics_t_uncond(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.Proj = nn.Sequential(
                                  nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128))
        self.mlp2 = nn.Sequential(nn.Linear(128,128),
                                  nn.SiLU(),
                                  nn.Linear(128,128),
                                  )
        self.condition_time = condition_time
        self.out = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )
    def forward(self, x,t, noidata=None, batch=None):
        noitarget = noidata

        temb = TimestepEmbedding(t,128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)
        target_hidden = target_hidden+temb
        output = self.out(target_hidden)
        output = noitarget-output
        output = output.squeeze(1)
        return output

class Dynamics_t_uncond_deeper(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.allfirst = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.Proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.down_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.mlp2 = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 256),
                                  nn.SiLU(),
                                  nn.Linear(256, 128),
                                  )

        self.condition_time = condition_time
        self.aftertime = nn.Sequential(nn.Linear(128, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 512),
                                       nn.SiLU(),
                                       nn.Linear(512, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 128),
                                       )

        self.aftertime2 = nn.Sequential(nn.Linear(128, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 512),
                                       nn.SiLU(),
                                       nn.Linear(512, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 128),
                                       )
        self.out = nn.Sequential(nn.Linear(128, 128),
                                 )
    def forward(self,t, noidata=None, batch=None):
        noitarget = noidata
        transed_target = self.allfirst(noitarget)
        temb = TimestepEmbedding(t,128),
        transed_target = transed_target+noitarget
        target_hidden = self.mlp2(transed_target)
        target_hidden = target_hidden+transed_target
        temb = temb[0]
        temb = self.Proj(temb)
        target_hidden = torch.cat((target_hidden,temb),dim=1)
        target_hidden = self.down_proj(target_hidden)
        target_hidden = target_hidden + temb
        output = self.aftertime(target_hidden)
        output = output+target_hidden
        output2 = self.aftertime2(output)
        output = output2 + output
        output = self.out(output)
        output = noitarget-output
        return output


class Dynamics_t_uncond_verydeep(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.ini = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.allfirst = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 192),
            nn.SiLU(),
            nn.Linear(192, 256),
            nn.SiLU(),
            nn.Linear(256, 320),
            nn.SiLU(),
            nn.Linear(320, 384),
            nn.SiLU(),
            nn.Linear(384, 448),
            nn.SiLU(),
            nn.Linear(448, 512),
        )
        self.Proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 192),
            nn.SiLU(),
            nn.Linear(192, 256),
            nn.SiLU(),
            nn.Linear(256, 320),
            nn.SiLU(),
            nn.Linear(320, 384),
            nn.SiLU(),
            nn.Linear(384, 448),
            nn.SiLU(),
            nn.Linear(448, 512),
        )
        self.ident = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )

        self.down_proj = nn.Sequential(
            nn.Linear(1024, 896),
            nn.SiLU(),
            nn.Linear(896, 768),
            nn.SiLU(),
            nn.Linear(768, 640),
            nn.SiLU(),
            nn.Linear(640, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),

        )

        self.condition_time = condition_time
        self.aftertime = nn.Sequential(nn.Linear(512, 384),
                                       nn.SiLU(),
                                       nn.Linear(384, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 196),
                                       nn.SiLU(),
                                       nn.Linear(196, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 128),
                                       )

        self.aftertime2 = nn.Sequential(nn.Linear(128, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 512),
                                       nn.SiLU(),
                                       nn.Linear(512, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 128),
                                        nn.SiLU(),
                                        nn.Linear(128, 128),
                                        nn.SiLU(),
                                        nn.Linear(128, 128),
                                        nn.SiLU(),
                                        nn.Linear(128, 128),
                                       )
        self.aftertime3 = nn.Sequential(nn.Linear(128, 128),
                                        nn.SiLU(),
                                        nn.Linear(128, 256),
                                        nn.SiLU(),
                                        nn.Linear(256, 512),
                                        nn.SiLU(),
                                        nn.Linear(512, 256),
                                        nn.SiLU(),
                                        nn.Linear(256, 128),
                                        nn.SiLU(),
                                        nn.Linear(128, 128),
                                        nn.SiLU(),
                                        nn.Linear(128, 128),
                                        nn.SiLU(),
                                        nn.Linear(128, 128),
                                        )

        self.out = nn.Sequential(nn.Linear(128, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 256),
                                 nn.SiLU(),
                                 nn.Linear(256, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 128),
                                 nn.SiLU(),
                                 nn.Linear(128, 128),
                                 )
    def forward(self,t, noidata=None, batch=None):
        noitarget = noidata
        transed_target = self.ini(noitarget)

        transed_target = transed_target+noitarget
        target_hidden = self.allfirst(transed_target)

        temb = TimestepEmbedding(t, 128),
        temb = temb[0]
        temb2 = self.Proj(temb)

        target_hidden = torch.cat((target_hidden,temb2),dim=1)
        target_hidden = self.down_proj(target_hidden)
        target_hidden = target_hidden + temb2
        target_hidden2 = self.ident(target_hidden)
        target_hidden = target_hidden2+target_hidden
        output = self.aftertime(target_hidden)
        output = output+temb
        output2 = self.aftertime2(output)
        output = output2 + output
        output3 = self.aftertime3(output)
        output = output3+output
        output = self.out(output)

        output = noitarget-output
        return output

class Dynamics_t_uncond_samp(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.Proj = nn.Sequential(
                                  nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128))
        self.mlp2 = nn.Sequential(nn.Linear(128,128),
                                  nn.SiLU(),
                                  nn.Linear(128,128),
                                  )

        self.condition_time = condition_time
        self.out = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )
    def forward(self,t,noidata=None,samp=True, look=1,num_samples=None, fresh_noise=None):
        noitarget = noidata
        temb = TimestepEmbedding(t, 128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)
        target_hidden = target_hidden + temb
        output = self.out(target_hidden)

        output = noitarget-output
        output = output.squeeze(1)
        return output

class Dynamics_t(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.mlp = nn.Linear(128,128)
        self.Proj = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128))
        self.mlp2 = nn.Sequential(nn.Linear(128,128),
                                  nn.SiLU(),
                                  nn.Linear(128,128),
                                  )
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
        self.condition_time = condition_time
        self.out = nn.Sequential(nn.Linear(256, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )
        self.conhidadj = nn.Linear(28, 128)
    def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):

        condition_x = condition_x
        condition_pos = condition_pos

        batch = batch
        target = data.target
        noitarget = noidata

        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        noitarget = noitarget.view(bs,-1)
        noitarget=noitarget.squeeze(1)
        temb = TimestepEmbedding(t, 128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)
        target_hidden = target_hidden + temb
        target_hidden=self.mlp(target_hidden)
        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_cond, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
        output = self.out(target_merged)
        return output


class Dynamics_t_esm_deeper(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.allfirst = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.Proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.down_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.mlp2 = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 256),
                                  nn.SiLU(),
                                  nn.Linear(256, 128),
                                  )

        self.condition_time = condition_time
        self.aftertime = nn.Sequential(nn.Linear(128, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 512),
                                       nn.SiLU(),
                                       nn.Linear(512, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 128),
                                       )

        self.aftertime2 = nn.Sequential(nn.Linear(128, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 512),
                                       nn.SiLU(),
                                       nn.Linear(512, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 128),
                                       )
        self.out = nn.Sequential(nn.Linear(128, 128),
                                 )

        self.cond_proj = nn.Sequential(nn.Linear(320, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 128),
                                       )
        self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
        self.afterattention = nn.Sequential(nn.Linear(256, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 512),
                                       nn.SiLU(),
                                       nn.Linear(512, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 128),
                                       )
    def forward(self,t, noidata=None, batch=None, esm_cond=None,mask=None):
        noitarget = noidata
        transed_target = self.allfirst(noitarget)
        temb = TimestepEmbedding(t,128),
        transed_target = transed_target+noitarget
        target_hidden = self.mlp2(transed_target)
        target_hidden = target_hidden+transed_target
        temb = temb[0]
        temb = self.Proj(temb)
        target_hidden = torch.cat((target_hidden,temb),dim=1)
        target_hidden = self.down_proj(target_hidden)
        target_hidden = target_hidden + temb
        output = self.aftertime(target_hidden)

        if len((esm_cond))<100:
            esm_cond = esm_cond.repeat(200,1,1)
            mask = mask.repeat(200, 1)
        esm_cond = self.cond_proj(esm_cond)

        mask = mask.unsqueeze(1).unsqueeze(2)
        target_cond, attention = self.attention_model(target_hidden, esm_cond, esm_cond, mask)
        target_mergerd = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
        target_hidden = self.afterattention(target_mergerd)
        output = output+target_hidden
        output2 = self.aftertime2(output)
        output = output2 + output
        output = self.out(output)
        output = noitarget-output
        return output

class Dynamics_t_esm(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.mlp = nn.Linear(128,128)
        self.Proj = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128))
        self.mlp2 = nn.Sequential(nn.Linear(128,128),
                                  nn.SiLU(),
                                  nn.Linear(128,128),
                                  )

        self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
        self.condition_time = condition_time
        self.out = nn.Sequential(nn.Linear(256, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )
        self.conhidadj = nn.Linear(28, 128)
    def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):

        condition_x = condition_x
        condition_pos = condition_pos

        batch = batch
        target = data.target
        noitarget = noidata

        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        noitarget = noitarget.view(bs,-1)
        noitarget=noitarget.squeeze(1)

        temb = TimestepEmbedding(t, 128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)
        target_hidden = target_hidden + temb
        target_hidden=self.mlp(target_hidden)
        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_cond, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)

        output = self.out(target_merged)

        return output

class Dynamics_t_samp(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.Proj = nn.Sequential(
                                  nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128))
        self.mlp2 = nn.Sequential(nn.Linear(128,128),
                                  nn.SiLU(),
                                  nn.Linear(128,128),
                                  )
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)

        self.condition_time = condition_time
        self.out = nn.Sequential(nn.Linear(256, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )
        self.conhidadj = nn.Linear(28, 128)
        self.mlp=nn.Linear(128,128)

    def set_value(self, cond,ma):
        self.cond = cond
        self.ma= ma

    def forward(self, data, condition_x,condition_pos, noidata, batch, t=None, samp = True, look=1,num_samples=None,fresh_noise=None):

        batch = batch
        noitarget = noidata
        print("now look",look)
        bs = max(batch) + 1
        print("bs:",bs)
        bs = bs*num_samples

        time_1=datetime.now()
        temb = TimestepEmbedding(t, 128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)
        target_hidden = target_hidden + temb
        target_hidden = self.mlp(target_hidden)
        time_2=datetime.now()
        time_3=datetime.now()
        if look == 0:
            condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
            condition_hidden = self.conhidadj(condition_hidden)
            condition_dense, mask = to_dense_batch(condition_hidden, batch)
            condition_dense = condition_dense.repeat(100, 1, 1)
            mask = mask.repeat(100, 1)
            mask = mask.unsqueeze(1).unsqueeze(2)
            self.set_value(condition_dense, mask)
        time_4=datetime.now()
        time_5 = datetime.now()
        target_cond, attention = self.attention_model(target_hidden, self.cond, self.cond, self.ma)
        time_6 = datetime.now()
        time_7 = datetime.now()
        target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
        output = self.out(target_merged)

        output = noitarget-output
        time_8 = datetime.now()

        return output
        
class Dynamics_egnn(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)

        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time
        self.out = nn.Linear(hid_dim, target_dim)
        self.conhidadj = nn.Linear(28, 64)
        self.deae = AE()

    def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):
        condition_x = condition_x
        condition_pos = condition_pos

        batch = batch
        target = data.target
        noitarget = noidata

        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        if self.condition_time:
           if np.prod(t.size()) == 1:
               h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
           target_with_time = torch.cat([noitarget, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)

        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        target = target.unsqueeze(1)
        error = target - output
        error1 = error.squeeze(1)
        output1 = output.squeeze(1)
        if samp:
            return output1
        elif not samp:
            return error1

class Dynamics_revive(nn.Module):
        def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                     n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
                     sampling=False):
            super().__init__()
            self.mlp = nn.Linear(target_dim + 1 if condition_time else target_dim, 128)
            self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
            self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
            self.condition_time = condition_time
            self.out = nn.Sequential(nn.Linear(256, target_dim),
                                     nn.ReLU(),
                                     nn.Linear(128,128))
            self.conhidadj = nn.Linear(28, 128)

        def forward(self, data, t, condition_x=None, condition_pos=None, noidata=None, batch=None, samp=False):
            target = data.target
            num_nodes = data.nodes
            bs = max(batch) + 1
            target = target.view(bs, -1)
            if self.condition_time:
                if np.prod(t.size()) == 1:
                    h_time = torch.empty_like(noidata[:, 0:1]).fill_(t.item())
                else:
                    h_time = t.view(bs, 1).repeat(1, 1)
                    h_time = h_time.view(bs, 1)
                target_with_time = torch.cat([noidata, h_time], dim=1)
                target_hidden = self.mlp(target_with_time)
            else:
                target_hidden = self.mlp(noidata)

            condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
            condition_hidden = self.conhidadj(condition_hidden)
            condition_dense, mask = to_dense_batch(condition_hidden, batch)
            mask = mask.unsqueeze(1).unsqueeze(2)
            target_cond, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
            target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
            output = self.out(target_merged)

            error = noidata - output
            return error

if __name__=='__main__':
    from torch_geometric.data import DataLoader
    from crossdock_dataset import PocketLigandPairDataset
