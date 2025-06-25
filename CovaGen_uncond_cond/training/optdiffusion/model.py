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
from optdiffusion.encdec import AE
import torch as th
import math
import esm
from datetime import datetime, date

# Should view the mods in gaussian_diffusion and mods here as two seperate phases isn't it? In gaussian the mods used
# to be the repeatance of noise and t, which is no different from what we do to noise now. so we don't need to modify mod
# used here? 3.2

# Now imma do some change to the model...

class Dynamics_revive_samp(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
                 sampling=False):
        super().__init__()
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        self.mlp = nn.Linear(target_dim + 1 if condition_time else target_dim, 128)
        # self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
        self.condition_time = condition_time  # 这个time是干嘛的？
        self.out = nn.Sequential(nn.Linear(256, target_dim),
                                 nn.ReLU(),
                                 nn.Linear(128, 128))  # 再过一个linear 64-128dim输出。
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
        bs = bs*num_samples # mod.!
        if self.condition_time:
           if np.prod(t.size()) == 1:
               # t is the same for all elements in batch.
               h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
               print("h_time shape:",h_time.shape)
               #h_time = h_time.repeat(2,1)
           target_with_time = torch.cat([noitarget, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)
        #print(len(target_hidden))
        if look == 0:
            condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
            condition_hidden = self.conhidadj(condition_hidden)
            condition_dense, mask = to_dense_batch(condition_hidden, batch)
            # condition_dense = condition_dense.view(100,654*64)
            # with open("/workspace/codeeval/condition.pkl",'wb')as f:
            #     pickle.dump(condition_dense[:,:50],f)
            print("orishape of mask",mask.shape)
            condition_dense=condition_dense.repeat(num_samples,1,1) # mod./
            mask=mask.repeat(num_samples,1) # mod.
            mask = mask.unsqueeze(1).unsqueeze(2)
            self.set_value(condition_dense,mask)
        #if look == 0:
        #    target_hidden=target_hidden.repeat(2,1)
        # print(condition_dense.shape)

        #print(mask.shape)
        #print("unsqed mask:", mask.shape)
        target_cond, attention = self.attention_model(target_hidden, self.cond, self.cond, self.ma)
        target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
        output = self.out(target_merged)
        #target = target.unsqueeze(1)
        # output1 = output.squeeze(1)
        error = noidata - output
        return error

class Dynamics(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Linear(hid_dim, target_dim)  # 再过一个linear 64-128dim输出。
        self.conhidadj = nn.Linear(28, 64)
    def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):
        # TODO: forward(self, condition_x,condition_pos, noise_target,batch,t=None, samp = False)
        target = data.target  # 此为原target。
        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        # noitarget = noitarget.view(bs,-1)
        if self.condition_time:
           if np.prod(t.size()) == 1:
               # t is the same for all elements in batch.
               h_time = torch.empty_like(noidata[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
           target_with_time = torch.cat([noidata, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noidata)  # TODO:input error

        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        # we got batch here. Just think about the trainning progress, when training how this model add conditional information
        # to the vector: we got 100k training data. suppose that the batchsize is 128, which means that we choose 128
        # different entries each time. An entry is made of
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        # target = target.unsqueeze(1)
        # error = target - output
        noidata = noidata.unsqueeze(1)#
        error = noidata - output
        print("error's shape:",error.shape)
        error = error.squeeze(1)
        print("squeezed shape:",error.shape)
        # output1 = output.squeeze(1)
        return error

class Dynamics_samp(nn.Module):  # 首先明确这个部分对应了ldm模型中的哪个位置，这样才好嵌入
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Linear(hid_dim, target_dim)  # 再过一个linear 64-128dim输出。
        self.conhidadj = nn.Linear(28, 64)
        
    def forward(self, data, condition_x,condition_pos, noidata, batch, t=None, samp = True, look=1,num_samples=None,fresh_noise=None):
        # TODO: forward(self, condition_x,condition_pos, noise_target,batch,t=None, samp = False)
        condition_x = condition_x  # 注意下这个dottable在哪儿实现的捏？下面这些属性难道是torch_geom里的dataset有这些属性？
        condition_pos = condition_pos

        batch = batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
        target = data.target  # 此为原target。
        noitarget = noidata


        #print("samp shape of noi:", noidata.shape)

        # if samp:    # This needs some alters...
        #     noitarget = noidata.squeeze(0)

        #print("noised length:", len(noitarget))
        #print("is noised allright?:", noitarget)
        #print("now look",look)
        num_nodes = data.nodes
        bs = max(batch) + 1
        #print("bs:",bs)
        bs = bs*num_samples # mod.!
        # target = target.view(bs, -1)
        # noitarget = noitarget.view(bs,-1)
        #print("viewed noitarget:", len(noitarget))
        if self.condition_time:
           if np.prod(t.size()) == 1:
               # t is the same for all elements in batch.
               h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
               #print("h_time shape:",h_time.shape)
               #h_time = h_time.repeat(2,1)
           target_with_time = torch.cat([noitarget, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)  # TODO:input error
        #print(len(target_hidden))

        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        #print("orishape of mask",mask.shape)
        condition_dense=condition_dense.repeat(num_samples,1,1) # mod./
        mask=mask.repeat(num_samples,1) # mod.
        #if look == 0:
        #    target_hidden=target_hidden.repeat(2,1)
        # print(condition_dense.shape)
        #print("shape de tar_hidden",target_hidden.shape)
        #print(mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(2)
        #print("unsqed mask:", mask.shape)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        # 疑问在这儿，未修改的target是batchsize*ldim的形状，而out是batchsize*1*ldim的形状，导致减出来一个别的形状的东西
        # 中间是后加的一句
        #target = target.unsqueeze(1)
        output1 = output.squeeze(1)
        error = noidata - output1
        # TODO:error不减直接输出试一下
        if samp:
            #print("nowwesample")
            # return output1
            return error
        elif not samp:
            return bs

class Dynamics_samp2(nn.Module):  # 首先明确这个部分对应了ldm模型中的哪个位置，这样才好嵌入
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Linear(hid_dim, target_dim)  # 再过一个linear 64-128dim输出。
        self.conhidadj = nn.Linear(28, 64)
        
    def forward(self, data, condition_x,condition_pos, noidata, batch, t=None, samp = True, look=1,num_samples=None,fresh_noise=None):
        # TODO: forward(self, condition_x,condition_pos, noise_target,batch,t=None, samp = False)
        condition_x = condition_x  # 注意下这个dottable在哪儿实现的捏？下面这些属性难道是torch_geom里的dataset有这些属性？
        condition_pos = condition_pos

        batch = batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
        target = data.target  # 此为原target。
        noitarget = noidata


        #print("samp shape of noi:", noidata.shape)

        # if samp:    # This needs some alters...
        #     noitarget = noidata.squeeze(0)

        #print("noised length:", len(noitarget))
        #print("is noised allright?:", noitarget)
        print("now look",look)
        num_nodes = data.nodes
        bs = max(batch) + 1
        print("bs:",bs)
        bs = bs*num_samples # mod.!
        # target = target.view(bs, -1)
        # noitarget = noitarget.view(bs,-1)
        #print("viewed noitarget:", len(noitarget))
        if self.condition_time:
           if np.prod(t.size()) == 1:
               # t is the same for all elements in batch.
               h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
               print("h_time shape:",h_time.shape)
               #h_time = h_time.repeat(2,1)
           target_with_time = torch.cat([noitarget, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)  # TODO:input error
        #print(len(target_hidden))

        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        print("orishape of mask",mask.shape)
        condition_dense=condition_dense.repeat(num_samples,1,1) # mod./
        mask=mask.repeat(num_samples,1) # mod.
        #if look == 0:
        #    target_hidden=target_hidden.repeat(2,1)
        # print(condition_dense.shape)
        print("shape de tar_hidden",target_hidden.shape)
        #print(mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(2)
        #print("unsqed mask:", mask.shape)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        # 疑问在这儿，未修改的target是batchsize*ldim的形状，而out是batchsize*1*ldim的形状，导致减出来一个别的形状的东西
        # 中间是后加的一句
        #target = target.unsqueeze(1)
        output1 = output.squeeze(1)
        error = output1-noidata
        # TODO:error不减直接输出试一下
        if samp:
            print("nowwesample")
            # return output1
            return error
        elif not samp:
            return bs

class Dynamics_nocond(nn.Module):  # 首先明确这个部分对应了ldm模型中的哪个位置，这样才好嵌入
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
                 sampling=False):
        super().__init__()
        self.condition_time = condition_time
        self.mlp = nn.Linear(128*65 + 1 if condition_time else target_dim, 1024) # 这个time是干嘛的？
        self.mlp2 = nn.Sequential(nn.Linear(1024, 512),
                                  nn.SiLU(),
                                  nn.Linear(512, 1024),
                                  )
        self.SiLU = nn.SiLU()
        self.out = nn.Linear(1024, 128*65)  # 再过一个linear 64-128dim输出。

    def forward(self, x,t, batch=None):#
        bs = batch
        noitarget = x
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, 1)
                h_time = h_time.view(bs, 1)
                print("h_time shape:", h_time.shape)
                # h_time = h_time.repeat(2,1)
            target_with_time = torch.cat([noitarget, h_time], dim=1)
            target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)  # TODO:input error
        target_hidden = self.SiLU(target_hidden)
        target_hidden = self.mlp2(target_hidden)
        print("shape de tar_hidden", target_hidden.shape)
        output = self.out(target_hidden)
        output1 = output.squeeze(1)
        error = noitarget - output1
        # error = error.view(-1,128,65)
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
        #target = data.target
        #bs = max(batch) + 1
        #target = target.view(bs, -1)
        #batch = batch
        noitarget = noidata
        bs = 10000
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, 1)
                h_time = h_time.view(bs, 1)
                #print("h_time shape:", h_time.shape)
                # h_time = h_time.repeat(2,1)
            target_with_time = torch.cat([noitarget, h_time], dim=1)
            target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)  # TODO:input error
        target_hidden = self.SiLU(target_hidden)
        target_hidden = self.mlp2(target_hidden)
        output = self.out(target_hidden)
        output1 = output.squeeze(1)
        error = noidata - output1
        return error

class Dynamics_noi(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Linear(hid_dim, target_dim)  # 再过一个linear 64-128dim输出。
        self.conhidadj = nn.Linear(28, 64)
    def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):
        # TODO: forward(self, condition_x,condition_pos, noise_target,batch,t=None, samp = False)
        condition_x = condition_x  # 注意下这个dottable在哪儿实现的捏？下面这些属性难道是torch_geom里的dataset有这些属性？
        condition_pos = condition_pos

        batch = batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
        target = data.target  # 此为原target。
        noitarget = noidata

        # if samp:    # This needs some alters...#
        #     noitarget = noidata.squeeze(0)

        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        # noitarget = noitarget.view(bs,-1)
        if self.condition_time:
           if np.prod(t.size()) == 1:
               # t is the same for all elements in batch.
               h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
           target_with_time = torch.cat([noitarget, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)  # TODO:input error


        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        # we got batch here. Just think about the trainning progress, when training how this model add conditional information
        # to the vector: we got 100k training data. suppose that the batchsize is 128, which means that we choose 128
        # different entries each time. An entry is made of
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        # target = target.unsqueeze(1)
        # error = target - output
        # error1 = error.squeeze(1)
        output1 = output.squeeze(1)
        return output1


class Dynamics_noi_samp(nn.Module):  # 首先明确这个部分对应了ldm模型中的哪个位置，这样才好嵌入
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
                 sampling=False):
        super().__init__()
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        self.mlp = nn.Linear(target_dim + 1 if condition_time else target_dim, hid_dim)
        # self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time  # 这个time是干嘛的？
        self.out = nn.Linear(hid_dim, target_dim)  # 再过一个linear 64-128dim输出。
        self.conhidadj = nn.Linear(28, 64)

    def forward(self, data, condition_x, condition_pos, noidata, batch, t=None, samp=True, look=1, num_samples=None,
                fresh_noise=None):
        # TODO: forward(self, condition_x,condition_pos, noise_target,batch,t=None, samp = False)
        condition_x = condition_x  # 注意下这个dottable在哪儿实现的捏？下面这些属性难道是torch_geom里的dataset有这些属性？
        condition_pos = condition_pos

        batch = batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
        target = data.target  # 此为原target。#
        noitarget = noidata

        print("now look", look)
        num_nodes = data.nodes
        bs = max(batch) + 1
        print("bs:", bs)
        bs = bs * num_samples  # mod.!
        # target = target.view(bs, -1)
        # noitarget = noitarget.view(bs,-1)
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, 1)
                h_time = h_time.view(bs, 1)
                print("h_time shape:", h_time.shape)
                # h_time = h_time.repeat(2,1)
            target_with_time = torch.cat([noitarget, h_time], dim=1)
            target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)  # TODO:input error
        # print(len(target_hidden))

        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        print("orishape of mask", mask.shape)
        condition_dense = condition_dense.repeat(num_samples, 1, 1)  # mod./
        mask = mask.repeat(num_samples, 1)  # mod.
        # if look == 0:
        #    target_hidden=target_hidden.repeat(2,1)
        # print(condition_dense.shape)
        print("shape de tar_hidden", target_hidden.shape)
        # print(mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(2)
        # print("unsqed mask:", mask.shape)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        # target = target.unsqueeze(1)
        output1 = output.squeeze(1)
        return output1
        # if samp:
        #     print("nowwesample")
        #     # return output1
        #     return output1
        # elif not samp:
        #     return bs

def TimestepEmbedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
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
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        # self.mlp2 = nn.Linear(target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )  # 再过一个linear 64-128dim输出。
    def forward(self, x,t, noidata=None, batch=None):
        noitarget = noidata
        # if self.condition_time:
        #    if np.prod(t.size()) == 1:
        #        # t is the same for all elements in batch.
        #        h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
        #    else:
        #        h_time = t.view(bs, 1).repeat(1, 1)
        #        h_time = h_time.view(bs , 1)
        #    target_with_time = torch.cat([noitarget, h_time], dim=1)
        #    target_hidden = self.mlp(target_with_time)
        # else:
        #     target_hidden = self.mlp(noitarget)
        temb = TimestepEmbedding(t,128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)
        target_hidden = target_hidden+temb
        output = self.out(target_hidden)
        # target = target.unsqueeze(1)
        # error = target - output
        # error1 = error.squeeze(1)
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

        self.condition_time = condition_time  # 这个time是干嘛的？
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
                                 # nn.SiLU(),
                                 # nn.Linear(128, 256),
                                 # nn.SiLU(),
                                 # nn.Linear(256, 128),
                                 )
    def forward(self,t, noidata=None, batch=None):
        noitarget = noidata
        transed_target = self.allfirst(noitarget)
        # if self.condition_time:
        #    if np.prod(t.size()) == 1:
        #        # t is the same for all elements in batch.
        #        h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
        #    else:#
        #        h_time = t.view(bs, 1).repeat(1, 1)
        #        h_time = h_time.view(bs , 1)
        #    target_with_time = torch.cat([noitarget, h_time], dim=1)
        #    target_hidden = self.mlp(target_with_time)
        # else:
        #     target_hidden = self.mlp(noitarget)
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
        # layer_outputs = []
        output = self.out(output)
        # target = target.unsqueeze(1)
        # error = target - output
        # error1 = error.squeeze(1)
        output = noitarget-output # 11.12在train完不减后发现不行，又打开了减，准备train一个更复杂的模型。后续：还行# 11.13 1：46 又不减了，train一个3000steps的模型。
        # 11.14 15:54,21减，train selfies的
        # output = output.squeeze(1)
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
        # self.mlp2 = nn.Sequential(nn.Linear(128, 128),
        #                           nn.SiLU(),
        #                           nn.Linear(128, 256),
        #                           nn.SiLU(),
        #                           nn.Linear(256, 128),
        #                           )

        self.condition_time = condition_time  # 这个time是干嘛的？
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
        transed_target = self.ini(noitarget) # x 128->128
        # if self.condition_time:
        #    if np.prod(t.size()) == 1:
        #        # t is the same for all elements in batch.
        #        h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
        #    else:#
        #        h_time = t.view(bs, 1).repeat(1, 1)
        #        h_time = h_time.view(bs , 1)
        #    target_with_time = torch.cat([noitarget, h_time], dim=1)
        #    target_hidden = self.mlp(target_with_time)
        # else:
        #     target_hidden = self.mlp(noitarget)

        transed_target = transed_target+noitarget # x = tx+x
        target_hidden = self.allfirst(transed_target) # x 128->512
        # target_hidden = target_hidden+transed_target

        temb = TimestepEmbedding(t, 128), # temb 128
        temb = temb[0]
        temb2 = self.Proj(temb) # temb2 512

        target_hidden = torch.cat((target_hidden,temb2),dim=1) # x = x cat time 512cat512 -> 1024
        target_hidden = self.down_proj(target_hidden) # x 1024->512
        target_hidden = target_hidden + temb2 # x + temb2
        target_hidden2 = self.ident(target_hidden) # x 512->512
        target_hidden = target_hidden2+target_hidden
        output = self.aftertime(target_hidden) # x 512->128
        output = output+temb # x + temb
        output2 = self.aftertime2(output) # X 128->512->128
        output = output2 + output
        output3 = self.aftertime3(output)
        output = output3+output
        # layer_outputs = []
        output = self.out(output)
        # target = target.unsqueeze(1)
        # error = target - output
        # error1 = error.squeeze(1)
        output = noitarget-output # 11.12在train完不减后发现不行，又打开了减，准备train一个更复杂的模型。
        # output = output.squeeze(1)
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
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        # self.mlp2 = nn.Linear(target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )  # 再过一个linear 64-128dim输出。
    def forward(self,t,noidata=None,samp=True, look=1,num_samples=None, fresh_noise=None):
        noitarget = noidata
        # if self.condition_time:
        #    if np.prod(t.size()) == 1:
        #        # t is the same for all elements in batch.
        #        h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
        #    else:
        #        h_time = t.view(bs, 1).repeat(1, 1)
        #        h_time = h_time.view(bs , 1)
        #    target_with_time = torch.cat([noitarget, h_time], dim=1)
        #    target_hidden = self.mlp(target_with_time)
        # else:
        #     target_hidden = self.mlp(noitarget)
        temb = TimestepEmbedding(t, 128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)
        target_hidden = target_hidden + temb
        output = self.out(target_hidden)
        # target = target.unsqueeze(1)
        # error = target - output
        # error1 = error.squeeze(1)
        output = noitarget-output
        output = output.squeeze(1)
        return output

class Dynamics_t(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()#
        self.mlp = nn.Linear(128,128)
        self.Proj = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128))
        self.mlp2 = nn.Sequential(nn.Linear(128,128),
                                  nn.SiLU(),
                                  nn.Linear(128,128),
                                  )
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        # self.mlp2 = nn.Linear(target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
        # self.attention_model2 = MultiHeadAttentionLayer(128, n_heads, dropout, device)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Sequential(nn.Linear(256, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )  # 再过一个linear 64-128dim输出。
        # self.out2 = nn.Sequential(nn.Linear(256, 128),
        #                           nn.SiLU(),
        #                           nn.Linear(128, 128),
        #                           )  # 再过一个linear 64-128dim输出。
        self.conhidadj = nn.Linear(28, 128)#
    def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):

        condition_x = condition_x  # 注意下这个dottable在哪儿实现的捏？下面这些属性难道是torch_geom里的dataset有这些属性？
        condition_pos = condition_pos

        batch = batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
        target = data.target  # 此为原target。
        noitarget = noidata

        # if samp:    # This needs some alters...#
        #     noitarget = noidata.squeeze(0)

        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        noitarget = noitarget.view(bs,-1)
        noitarget=noitarget.squeeze(1)
        # if self.condition_time:
        #    if np.prod(t.size()) == 1:
        #        # t is the same for all elements in batch.
        #        h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
        #    else:
        #        h_time = t.view(bs, 1).repeat(1, 1)
        #        h_time = h_time.view(bs , 1)
        #    target_with_time = torch.cat([noitarget, h_time], dim=1)
        #    target_hidden = self.mlp(target_with_time)
        # else:
        #     target_hidden = self.mlp(noitarget)
        temb = TimestepEmbedding(t, 128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)#
        target_hidden = target_hidden + temb
        target_hidden=self.mlp(target_hidden)
        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        # we got batch here. Just think about the trainning progress, when training how this model add conditional information
        # to the vector: we got 100k training data. suppose that the batchsize is 128, which means that we choose 128
        # different entries each time. An entry is made of
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_cond, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)#

        #jia
        # target_merged = self.out(target_merged)
        # target_cond, attention = self.attention_model2(target_merged, condition_dense, condition_dense, mask)
        # target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)

        output = self.out(target_merged)
        # target = target.unsqueeze(1)
        # error = target - output
        # error1 = error.squeeze(1)
        # output = noitarget-output
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
        )#
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

        self.condition_time = condition_time  # 这个time是干嘛的？
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
                                 # nn.SiLU(),
                                 # nn.Linear(128, 256),
                                 # nn.SiLU(),
                                 # nn.Linear(256, 128),
                                 )

        #condition
        self.cond_proj = nn.Sequential(nn.Linear(320, 256),
                                       nn.SiLU(),
                                       nn.Linear(256, 128),
                                       nn.SiLU(),
                                       nn.Linear(128, 128),
                                       )
        #attention
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
        # if self.condition_time:
        #    if np.prod(t.size()) == 1:
        #        # t is the same for all elements in batch.
        #        h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
        #    else:#
        #        h_time = t.view(bs, 1).repeat(1, 1)
        #        h_time = h_time.view(bs , 1)
        #    target_with_time = torch.cat([noitarget, h_time], dim=1)
        #    target_hidden = self.mlp(target_with_time)

        # else:
        #     target_hidden = self.mlp(noitarget)
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

        ## sampling
        if len((esm_cond))<100:
            esm_cond = esm_cond.repeat(200,1,1)
            mask = mask.repeat(200, 1)
        esm_cond = self.cond_proj(esm_cond)

        mask = mask.unsqueeze(1).unsqueeze(2)
        target_cond, attention = self.attention_model(target_hidden, esm_cond, esm_cond, mask)
        target_mergerd = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)  #
        target_hidden = self.afterattention(target_mergerd)
        output = output+target_hidden
        output2 = self.aftertime2(output)
        output = output2 + output
        # layer_outputs = []
        output = self.out(output)
        # target = target.unsqueeze(1)
        # error = target - output
        # error1 = error.squeeze(1)
        output = noitarget-output # 11.12在train完不减后发现不行，又打开了减，准备train一个更复杂的模型。后续：还行# 11.13 1：46 又不减了，train一个3000steps的模型。
        # 11.14 15:54,21减，train selfies的
        # output = output.squeeze(1)
        return output

class Dynamics_t_esm(nn.Module):
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()#
        self.mlp = nn.Linear(128,128)
        self.Proj = nn.Sequential(nn.Linear(128, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128))
        self.mlp2 = nn.Sequential(nn.Linear(128,128),
                                  nn.SiLU(),
                                  nn.Linear(128,128),
                                  )
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        # self.mlp2 = nn.Linear(target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        # self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
        # self.attention_model2 = MultiHeadAttentionLayer(128, n_heads, dropout, device)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Sequential(nn.Linear(256, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )  # 再过一个linear 64-128dim输出。
        # self.out2 = nn.Sequential(nn.Linear(256, 128),
        #                           nn.SiLU(),
        #                           nn.Linear(128, 128),
        #                           )  # 再过一个linear 64-128dim输出。
        self.conhidadj = nn.Linear(28, 128)#
    def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):

        condition_x = condition_x  # 注意下这个dottable在哪儿实现的捏？下面这些属性难道是torch_geom里的dataset有这些属性？
        condition_pos = condition_pos

        batch = batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
        target = data.target  # 此为原target。
        noitarget = noidata

        # if samp:    # This needs some alters...#
        #     noitarget = noidata.squeeze(0)

        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        noitarget = noitarget.view(bs,-1)
        noitarget=noitarget.squeeze(1)
        # if self.condition_time:
        #    if np.prod(t.size()) == 1:
        #        # t is the same for all elements in batch.
        #        h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
        #    else:
        #        h_time = t.view(bs, 1).repeat(1, 1)
        #        h_time = h_time.view(bs , 1)
        #    target_with_time = torch.cat([noitarget, h_time], dim=1)
        #    target_hidden = self.mlp(target_with_time)
        # else:
        #     target_hidden = self.mlp(noitarget)
        temb = TimestepEmbedding(t, 128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)#
        target_hidden = target_hidden + temb
        target_hidden=self.mlp(target_hidden)
        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        # we got batch here. Just think about the trainning progress, when training how this model add conditional information
        # to the vector: we got 100k training data. suppose that the batchsize is 128, which means that we choose 128
        # different entries each time. An entry is made of
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        mask = mask.unsqueeze(1).unsqueeze(2)
        target_cond, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)#

        #jia
        # target_merged = self.out(target_merged)
        # target_cond, attention = self.attention_model2(target_merged, condition_dense, condition_dense, mask)
        # target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)

        output = self.out(target_merged)
        # target = target.unsqueeze(1)
        # error = target - output
        # error1 = error.squeeze(1)
        # output = noitarget-output
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
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        # self.mlp2 = nn.Linear(target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)

        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Sequential(nn.Linear(256, 128),
                                  nn.SiLU(),
                                  nn.Linear(128, 128),
                                  )  # 再过一个linear 64-128dim输出。
        self.conhidadj = nn.Linear(28, 128)
        self.mlp=nn.Linear(128,128)

    def set_value(self, cond,ma):
        self.cond = cond
        self.ma= ma

    def forward(self, data, condition_x,condition_pos, noidata, batch, t=None, samp = True, look=1,num_samples=None,fresh_noise=None):

        # condition_x = condition_x  # 注意下这个dottable在哪儿实现的捏？下面这些属性难道是torch_geom里的dataset有这些属性？
        # condition_pos = condition_pos
        #
        # batch = batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
        # target = data.target  # 此为原target。
        # noitarget = noidata
        #
        # # if samp:    # This needs some alters...#
        # #     noitarget = noidata.squeeze(0)
        #
        # num_nodes = data.nodes
        # bs = max(batch) + 1
        # target = target.view(bs, -1)
        # noitarget = noitarget.view(bs,-1)
        # noitarget=noitarget.squeeze(1)

        batch = batch
        noitarget = noidata
        print("now look",look)
        bs = max(batch) + 1
        print("bs:",bs)
        bs = bs*num_samples # mod.!

        # if self.condition_time:
        #    if np.prod(t.size()) == 1:
        #        # t is the same for all elements in batch.
        #        h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
        #    else:
        #        h_time = t.view(bs, 1).repeat(1, 1)
        #        h_time = h_time.view(bs , 1)
        #    target_with_time = torch.cat([noitarget, h_time], dim=1)
        #    target_hidden = self.mlp(target_with_time)
        # else:
        #     target_hidden = self.mlp(noitarget)
        time_1=datetime.now()
        temb = TimestepEmbedding(t, 128),
        target_hidden = self.mlp2(noitarget)
        temb = temb[0]
        temb = self.Proj(temb)  #
        target_hidden = target_hidden + temb
        target_hidden = self.mlp(target_hidden)
        time_2=datetime.now()
        #print("time spent for timestepembedding:", (time_2 - time_1).microseconds)
        time_3=datetime.now()
        if look == 0:
            condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
            condition_hidden = self.conhidadj(condition_hidden)
            condition_dense, mask = to_dense_batch(condition_hidden, batch)
            condition_dense = condition_dense.repeat(100, 1, 1)  # mod./
            mask = mask.repeat(100, 1)  # mod.
            mask = mask.unsqueeze(1).unsqueeze(2)
            self.set_value(condition_dense, mask)
        time_4=datetime.now()
        #print("time spent for condition encoding:", (time_4-time_3).microseconds)
        time_5 = datetime.now()
        target_cond, attention = self.attention_model(target_hidden, self.cond, self.cond, self.ma)
        time_6 = datetime.now()
        #print("time spent for attention:", (time_6-time_5).microseconds)
        time_7 = datetime.now()
        target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
        output = self.out(target_merged)
        # target = target.unsqueeze(1)
        # error = target - output
        # error1 = error.squeeze(1)
        output = noitarget-output
        time_8 = datetime.now()
        #print("time spent for outputting:",(time_8-time_7).microseconds)
        return output
        
class Dynamics_egnn(nn.Module):  # 首先明确这个部分对应了ldm模型中的哪个位置，这样才好嵌入
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        # self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Linear(hid_dim, target_dim)  # 再过一个linear 64-128dim输出。
        self.conhidadj = nn.Linear(28, 64)
        self.deae = AE()

    def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):
        condition_x = condition_x  # 注意下这个dottable在哪儿实现的捏？下面这些属性难道是torch_geom里的dataset有这些属性？
        condition_pos = condition_pos

        batch = batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
        target = data.target  # 此为原target。
        noitarget = noidata

        #print("samp shape of noi:", noidata.shape)

        # if samp:    # This needs some alters...
        #     noitarget = noidata.squeeze(0)

        #print("noised length:", len(noitarget))
        #print("is noised allright?:", noitarget)

        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        # noitarget = noitarget.view(bs,-1)
        #print("viewed noitarget:", len(noitarget))
        #print("tsize:",np.prod(t.size()))
        if self.condition_time:
           if np.prod(t.size()) == 1:
               # t is the same for all elements in batch.
               h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
           else:
               h_time = t.view(bs, 1).repeat(1, 1)
               h_time = h_time.view(bs , 1)
           target_with_time = torch.cat([noitarget, h_time], dim=1)
           target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(noitarget)  # TODO:input error
        #print(len(target_hidden))
        #print("targethidden's shape",target_hidden.shape)

        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        #print("before rpt:",condition_dense.shape)
        #condition_dense = condition_dense.repeat(2,1,1)
        #print("after rpt:",condition_dense.shape)
        #print(condition_dense.shape)
        #print(mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(2)
        #print("unsqed mask:", mask.shape)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        # 疑问在这儿，未修改的target是batchsize*ldim的形状，而out是batchsize*1*ldim的形状，导致减出来一个别的形状的东西
        # 中间是后加的一句
        target = target.unsqueeze(1)
        # 1
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
            # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
            self.mlp = nn.Linear(target_dim + 1 if condition_time else target_dim, 128)
            # self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
            self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
            # EGNNEncoder这里是对pocket进行编码嵌入
            self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
            self.condition_time = condition_time  # 这个time是干嘛的？
            self.out = nn.Sequential(nn.Linear(256, target_dim),
                                     nn.ReLU(),
                                     nn.Linear(128,128))
            self.conhidadj = nn.Linear(28, 128)

        def forward(self, data, t, condition_x=None, condition_pos=None, noidata=None, batch=None, samp=False):
            # TODO: forward(self, condition_x,condition_pos, noise_target,batch,t=None, samp = False)
            target = data.target  # 此为原target。
            num_nodes = data.nodes
            bs = max(batch) + 1
            target = target.view(bs, -1)
            # noitarget = noitarget.view(bs,-1)
            if self.condition_time:
                if np.prod(t.size()) == 1:
                    # t is the same for all elements in batch.
                    h_time = torch.empty_like(noidata[:, 0:1]).fill_(t.item())
                else:
                    h_time = t.view(bs, 1).repeat(1, 1)
                    h_time = h_time.view(bs, 1)
                target_with_time = torch.cat([noidata, h_time], dim=1)
                target_hidden = self.mlp(target_with_time)
            else:
                target_hidden = self.mlp(noidata)  # TODO:input error

            condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
            # we got batch here. Just think about the trainning progress, when training how this model add conditional information
            # to the vector: we got 100k training data. suppose that the batchsize is 128, which means that we choose 128
            # different entries each time. An entry is made of
            condition_hidden = self.conhidadj(condition_hidden)
            condition_dense, mask = to_dense_batch(condition_hidden, batch)
            mask = mask.unsqueeze(1).unsqueeze(2)
            target_cond, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
            target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
            output = self.out(target_merged)
            # target = target.unsqueeze(1)
            # error = target - output
            # noidata = noidata.unsqueeze(1)  #
            error = noidata - output
            # error = output
            # error = error.squeeze(1)
            # output1 = output.squeeze(1)
            return error
# class Dynamics_ae(nn.Module):  # 首先明确这个部分对应了ldm模型中的哪个位置，这样才好嵌入
#     def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
#                  n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
#         super().__init__()
#         # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
#         self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
#         #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
#         self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
#         # EGNNEncoder这里是对pocket进行编码嵌入
#         self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
#         self.condition_time = condition_time    # 这个time是干嘛的？
#         self.out = nn.Linear(hid_dim, target_dim)  # 再过一个linear 64-128dim输出。
#         self.conhidadj = nn.Linear(28, 64)
#         self.deae = AE()
#
#     def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False):
#         # TODO: forward(self, condition_x,condition_pos, noise_target,batch,t=None, samp = False)
#         condition_x = condition_x  # 注意下这个dottable在哪儿实现的捏？下面这些属性难道是torch_geom里的dataset有这些属性？
#         condition_pos = condition_pos
#
#         batch = batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
#         target = data.target  # 此为原target。
#         noitarget = noidata
#
#         #print("samp shape of noi:", noidata.shape)
#
#         # if samp:    # This needs some alters...
#         #     noitarget = noidata.squeeze(0)
#
#         #print("noised length:", len(noitarget))
#         #print("is noised allright?:", noitarget)
#
#         num_nodes = data.nodes
#         bs = max(batch) + 1
#         target = target.view(bs, -1)
#         # noitarget = noitarget.view(bs,-1)
#         #print("viewed noitarget:", len(noitarget))
#         #print("tsize:",np.prod(t.size()))
#         if self.condition_time:
#            if np.prod(t.size()) == 1:
#                # t is the same for all elements in batch.
#                h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
#            else:
#                h_time = t.view(bs, 1).repeat(1, 1)
#                h_time = h_time.view(bs , 1)
#            target_with_time = torch.cat([noitarget, h_time], dim=1)
#            target_hidden = self.mlp(target_with_time)
#         else:
#             target_hidden = self.mlp(noitarget)  # TODO:input error
#         #print(len(target_hidden))
#         #print("targethidden's shape",target_hidden.shape)
#
#         condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
#         condition_hidden = self.conhidadj(condition_hidden)
#         condition_dense, mask = to_dense_batch(condition_hidden, batch)
#         #print("before rpt:",condition_dense.shape)
#         #condition_dense = condition_dense.repeat(2,1,1)
#         #print("after rpt:",condition_dense.shape)
#         #print(condition_dense.shape)
#         #print(mask.shape)
#         mask = mask.unsqueeze(1).unsqueeze(2)
#         #print("unsqed mask:", mask.shape)
#         target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
#         output_befae = self.out(target_merged)
#         output = self.deae(output_befae)
#         # 疑问在这儿，未修改的target是batchsize*ldim的形状，而out是batchsize*1*ldim的形状，导致减出来一个别的形状的东西
#         # 中间是后加的一句
#         target = target.unsqueeze(1)
#         # 1
#         error = target - output
#         error1 = error.squeeze(1)
#         output1 = output.squeeze(1)
#         if samp:
#             return output1
#         elif not samp:
#             return error1


if __name__=='__main__':
    from torch_geometric.data import DataLoader
    from crossdock_dataset import PocketLigandPairDataset

    device = torch.device('cuda:0')
    dataset = PocketLigandPairDataset('/workspace/stu/ltx/nt/dataset/dataset/',
                                      vae_path='/workspace/stu/ltx/nt/045_trans1x-128_zinc.ckpt',
                                      save_path='/workspace/stu/ltx/nt/dataset/dataset/processed/')
    loader = DataLoader(dataset, batch_size=2)

    criterion = nn.MSELoss()
    target = torch.tensor([[0] * 128, [1] * 128]).float().to(device)

    model = Dynamics(condition_dim=28, target_dim=128, hid_dim=64, condition_layer=3, n_heads=2,
                     condition_time=True).to(device)

    model_params = list(model.parameters())

    for batch in loader:
        batch = batch.to(device)
        print(batch)
        out = model(batch, t=torch.tensor(1))
        loss = criterion(out, target)
        # out.backward()
        grads = []
        grad_none = 0
        for para in model_params:
            if para.grad is None:
                grad_none += 1

        print(out)