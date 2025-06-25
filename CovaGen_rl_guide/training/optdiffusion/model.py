import pickle
from datetime import datetime

import torch
import os,sys
import math
import json
sys.path.append(os.path.dirname(sys.path[0]))
from torch import nn
from optdiffusion.EGNNEncoder_pyg import EGNNEncoder
from optdiffusion.MultiHeadAttentionLayer import MultiHeadAttentionLayer
import numpy as np
from torch import utils
from torch_geometric.utils import to_dense_batch
from dgg.models.encoders.schnet import SchNetEncoder



# Now imma do some change to the model...
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

	def forward(self, data, t, condition_x=None,condition_pos=None, noidata=None, batch=None, samp = False): # 回忆，dataloader中一个batch的batch属性存储的是什么？
		target = data.target
		bs = max(batch) + 1
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
		error = noidata-output
		error = error.squeeze(1)
		return error

class Dynamics_samp(nn.Module):  # 首先明确这个部分对应了ldm模型中的哪个位置，这样才好嵌入
	def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
				 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
		super().__init__()
		self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
		self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
		self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
		self.condition_time = condition_time
		self.out = nn.Linear(hid_dim, target_dim)
		self.conhidadj = nn.Linear(28, 64)

	def forward(self, batch, condition_x, condition_pos,noidata, t=None, samp=False):
		# batch = batch
		noitarget = noidata
		bs = 24
		num_samples=noidata.shape[0]
		# bs = bs*num_samples # mod.!，
		if self.condition_time:
			if np.prod(t.size()) == 1:

				h_time = torch.empty_like(noitarget[:, 0:1]).fill_(t.item())
			else:
				# h_time = t.view(bs, 1).repeat(1, 1)
				# h_time = h_time.view(bs , 1)
				h_time = t.reshape (-1, 1)

			target_with_time = torch.cat([noitarget, h_time], dim=1)
			target_hidden = self.mlp(target_with_time)
		else:
			target_hidden = self.mlp(noitarget)
		condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch.batch)
		condition_hidden = self.conhidadj(condition_hidden)
		condition_dense, mask = to_dense_batch(condition_hidden, batch.batch)
		# condition_dense=condition_dense.repeat(num_samples,1,1) # mod./
		# mask=mask.repeat(num_samples,1) # mod.
		mask = mask.unsqueeze(1).unsqueeze(2)
		target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
		output = self.out(target_merged)

		output1 = output.squeeze(1)
		error = noidata - output1
		return error



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
	freqs = torch.exp(
		-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
	).to(device=timesteps.device)
	args = timesteps[:, None].float() * freqs[None]
	embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
	if dim % 2:
		embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
	return embedding


class Dynamics_t_uncond(nn.Module):
	def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
				 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
				 sampling=False):
		super().__init__()
		self.Proj = nn.Sequential(
			nn.Linear(128, 128),
			nn.SiLU(),
			nn.Linear(128, 128))
		self.mlp2 = nn.Sequential(nn.Linear(128, 128),
								  nn.SiLU(),
								  nn.Linear(128, 128),
								  )

		self.condition_time = condition_time
		self.out = nn.Sequential(nn.Linear(128, 128),
								 nn.SiLU(),
								 nn.Linear(128, 128),
								 )
	def forward(self, x, t, noidata=None, batch=None):
		noitarget = noidata

		temb = TimestepEmbedding(t, 128),
		target_hidden = self.mlp2(noitarget)
		temb = temb[0]
		temb = self.Proj(temb)
		target_hidden = target_hidden + temb
		output = self.out(target_hidden)

		output = noitarget - output
		output = output.squeeze(1)
		return output


class Dynamics_t_uncond_samp(nn.Module):
	def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
				 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
				 sampling=False):
		super().__init__()
		self.Proj = nn.Sequential(
			nn.Linear(128, 128),
			nn.SiLU(),
			nn.Linear(128, 128))
		self.mlp2 = nn.Sequential(nn.Linear(128, 128),
								  nn.SiLU(),
								  nn.Linear(128, 128),
								  )

		self.condition_time = condition_time  # 这个time是干嘛的？
		self.out = nn.Sequential(nn.Linear(128, 128),
								 nn.SiLU(),
								 nn.Linear(128, 128),
								 )

	def forward(self, noidata, t=None, samp=True, look=1, num_samples=None, fresh_noise=None):
		noitarget = noidata

		temb = TimestepEmbedding(t, 128),
		target_hidden = self.mlp2(noitarget)
		temb = temb[0]
		temb = self.Proj(temb)
		target_hidden = target_hidden + temb
		output = self.out(target_hidden)

		output = noitarget - output
		output = output.squeeze(1)
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

        # sampling
        # if len((esm_cond))<100:
        #     esm_cond = esm_cond.repeat(100,1,1)
        #     mask = mask.repeat(100, 1)
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

class Dynamics_t(nn.Module): #
	def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
				 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
				 sampling=False):
		super().__init__()
		self.mlp = nn.Linear(128, 128)
		self.Proj = nn.Sequential(nn.Linear(128, 128),
								  nn.SiLU(),
								  nn.Linear(128, 128))
		self.mlp2 = nn.Sequential(nn.Linear(128, 128),
								  nn.SiLU(),
								  nn.Linear(128, 128),
								  )
		self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)

		self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
		self.condition_time = condition_time
		self.out = nn.Sequential(nn.Linear(256, 128),
								 nn.SiLU(),
								 nn.Linear(128, 128),
								 )
		self.conhidadj = nn.Linear(28, 128)  #

	def forward(self, batch, condition_x, condition_pos,noidata, t=None, samp=False):
		condition_x = condition_x
		condition_pos = condition_pos#

		batch = batch
		noitarget = noidata
		bs = 128
		# noitarget = noitarget.view(bs, -1)
		# noitarget = noitarget.squeeze(1)


		temb = TimestepEmbedding(t, 128),
		target_hidden = self.mlp2(noitarget)
		temb = temb[0]
		temb = self.Proj(temb)  #
		target_hidden = target_hidden + temb
		target_hidden = self.mlp(target_hidden)
		condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch.batch)
		condition_hidden = self.conhidadj(condition_hidden)
		condition_dense, mask = to_dense_batch(condition_hidden, batch.batch)
		mask = mask.unsqueeze(1).unsqueeze(2)
		target_cond, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
		target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
		output = self.out(target_merged)
		output = noitarget - output
		return output

	def get_network_parameters(self):
		return self

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

	def forward(self, data, condition_x,condition_pos, noidata, batch, t=None, samp = True, look=0,num_samples=None,fresh_noise=None):

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
		# bs = bs*num_samples # mod.!

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
		print("time spent for timestepembedding:", (time_2 - time_1).microseconds)
		time_3=datetime.now()
		if look == 0:
			condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
			condition_hidden = self.conhidadj(condition_hidden)
			condition_dense, mask = to_dense_batch(condition_hidden, batch)
			condition_dense = condition_dense.repeat(100, 1, 1)  # mod./
			mask = mask.repeat(100, 1)  # mod.
			mask = mask.unsqueeze(1).unsqueeze(2)
			self.set_value(condition_dense, mask)
			look+=1
		time_4=datetime.now()
		print("time spent for condition encoding:", (time_4-time_3).microseconds)
		time_5 = datetime.now()
		target_cond, attention = self.attention_model(target_hidden, self.cond, self.cond, self.ma)
		time_6 = datetime.now()
		print("time spent for attention:", (time_6-time_5).microseconds)
		time_7 = datetime.now()
		target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
		output = self.out(target_merged)
		# target = target.unsqueeze(1)
		# error = target - output
		# error1 = error.squeeze(1)
		output = noitarget-output
		time_8 = datetime.now()
		print("time spent for outputting:",(time_8-time_7).microseconds)
		return output

class Dynamics_t_samp2(nn.Module): #
	def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
				 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'),
				 sampling=False):
		super().__init__()
		self.Proj = nn.Sequential(
			nn.Linear(128, 128),
			nn.SiLU(),
			nn.Linear(128, 128))
		self.mlp2 = nn.Sequential(nn.Linear(128, 128),
								  nn.SiLU(),
								  nn.Linear(128, 128),
								  )
		self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
		self.attention_model = MultiHeadAttentionLayer(128, n_heads, dropout, device)
		self.condition_time = condition_time
		self.out = nn.Sequential(nn.Linear(256, 128),
								 nn.SiLU(),
								 nn.Linear(128, 128),
								 )
		self.conhidadj = nn.Linear(28, 128)
		self.mlp = nn.Linear(128, 128)
		self.pres = 0

	def set_value(self, cond, ma):
		self.cond = cond
		self.ma = ma

	def forward(self, data, condition_x, condition_pos, noidata, batch, t=None, samp=True, look=1, num_samples=None,
				fresh_noise=None):

		batch = batch
		noitarget = noidata
		num_samples=2000

		temb = TimestepEmbedding(t, 128),
		target_hidden = self.mlp2(noitarget)
		temb = temb[0]
		temb = self.Proj(temb)  #
		target_hidden = target_hidden + temb
		target_hidden = self.mlp(target_hidden)
		if self.pres == 0:
			condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
			condition_hidden = self.conhidadj(condition_hidden)
			condition_dense, mask = to_dense_batch(condition_hidden, batch)
			condition_dense = condition_dense.repeat(num_samples, 1, 1)  # mod./
			mask = mask.repeat(num_samples, 1)  # mod.
			mask = mask.unsqueeze(1).unsqueeze(2)
			self.set_value(condition_dense, mask)
			self.pres+=1
		target_cond, attention = self.attention_model(target_hidden, self.cond, self.cond, self.ma)

		#for dev
		# target_cond = target_hidden

		target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
		output = self.out(target_merged)
		output = noitarget - output
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
        # output = output.squeeze(1)
        return output

class simple_reg_Net_newt(torch.nn.Module):
	def __init__(self):
		super(simple_reg_Net_newt,self).__init__()
		self.go = nn.Linear(256,256)#
		self.mlp = nn.Sequential(
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.SiLU(),
			nn.Linear(128, 64),
			nn.SiLU(),
			nn.Linear(64, 16),
			nn.SiLU(),
			nn.Linear(16, 8),
			nn.SiLU(),
			nn.Linear(8,1),
		)
		self.time_embed = nn.Sequential(
			nn.Linear(128, 256),
			nn.SiLU(),
			nn.Linear(256, 128),
		)



	def forward(self, x, timesteps=None):
		bs = len(x)
		# if np.prod(timesteps.size()) == 1:
		# 	# t is the same for all elements in batch.
		# 	h_time = torch.empty_like(x[:, 0:1]).fill_(timesteps.item())
		# else:
		# 	h_time = timesteps.view(bs, 1).repeat(1, 1)
		# 	h_time = h_time.view(bs, 1)
		# # h_time = h_time.repeat(2,1)
		temb = TimestepEmbedding(timesteps, 128),
		h_time = self.time_embed(temb[0])
		x = torch.cat([x, h_time], dim=1)
		x = self.go(x)
		return self.mlp(x)

	def predict(self,x,timesteps=None):
		return self.forward(x,timesteps)

class xboost_scoring:
	def __init__(self):
		with open('/workspace/codes/emiddpm/toxity/xgboost_model_maccs.pkl', "rb") as model_file:
			loaded_model = pickle.load(model_file)
	def predict(self):
		vae = RNNAttn(load_fn=args.vae_path)
		
if __name__=='__main__':
	from torch_geometric.data import DataLoader
	from crossdock_dataset import PocketLigandPairDataset
#
# device = torch.device('cuda:0')
# dataset = PocketLigandPairDataset('/workspace/stu/ltx/nt/dataset/dataset/',
#                                   vae_path='/workspace/stu/ltx/nt/045_trans1x-128_zinc.ckpt',
#                                   save_path='/workspace/stu/ltx/nt/dataset/dataset/processed/')
# loader = DataLoader(dataset, batch_size=2)
#
# criterion = nn.MSELoss()
# target = torch.tensor([[0] * 128, [1] * 128]).float().to(device)
#
# model = Dynamics(condition_dim=28, target_dim=128, hid_dim=64, condition_layer=3, n_heads=2,
#                  condition_time=True).to(device)
#
# model_params = list(model.parameters())
#
# for batch in loader:
#     batch = batch.to(device)
#     print(batch)
#     out = model(batch, t=torch.tensor(1))
#     loss = criterion(out, target)
#     # out.backward()
#     grads = []
#     grad_none = 0
#     for para in model_params:
#         if para.grad is None:
#             grad_none += 1
#
#     print(out)