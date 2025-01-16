import torch
import numpy as np
from torch import nn#
import math
# from optdiffusion.EGNNEncoder_pyg import EGNNEncoder
from optdiffusion.MultiHeadAttentionLayer import MultiHeadAttentionLayer
import numpy as np
from torch import utils
from torch_geometric.utils import to_dense_batch#
from dgg.models.encoders.schnet import SchNetEncoder

def TimestepEmbedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class timeNet(torch.nn.Module):
	def __init__(self):
		super(timeNet, self).__init__()
		self.Proj = nn.Sequential(
			nn.Linear(128, 128),
			nn.SiLU(),
			nn.Linear(128, 128))
		self.mlp2 = nn.Sequential(nn.Linear(128, 128),
								  nn.SiLU(),
								  nn.Linear(128, 128),
								  )

		self.condition_time = True
		self.out = nn.Sequential(nn.Linear(128, 128),
								 nn.SiLU(),
								 nn.Linear(128, 128),
								 nn.SiLU(),
								 nn.Linear(128,64),
								 nn.SiLU(),
								 nn.Linear(64,32),
								 nn.SiLU(),
								 nn.Linear(32,16),
								 nn.SiLU(),
								 nn.Linear(16,2),
								 )

	def forward(self, noidata, timesteps=None):
		noitarget = noidata
		temb = TimestepEmbedding(timesteps, 128),
		target_hidden = self.mlp2(noitarget)
		temb = temb[0]
		temb = self.Proj(temb)
		target_hidden = target_hidden + temb
		output = self.out(target_hidden)
		return output


class condtimeNet(torch.nn.Module):
	def __init__(self):
		super(condtimeNet, self).__init__()
		self.mlp = nn.Linear(128, 128)
		self.Proj = nn.Sequential(nn.Linear(128, 128),
								  nn.SiLU(),
								  nn.Linear(128, 128))
		self.mlp2 = nn.Sequential(nn.Linear(128, 128),
								  nn.SiLU(),
								  nn.Linear(128, 128),
								  )
		self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=5)
		self.start = True
		self.attention_model = MultiHeadAttentionLayer(128, 2, 0.2, torch.device('cuda:0'))
		self.condition_time = True
		self.out = nn.Sequential(nn.Linear(256, 128),
								 nn.SiLU(),
								 nn.Linear(128, 128),
								 nn.SiLU(),
								 nn.Linear(128, 64),
								 nn.SiLU(),
								 nn.Linear(64, 32),
								 nn.SiLU(),
								 nn.Linear(32, 16),
								 nn.SiLU(),
								 nn.Linear(16, 2),
								 )
		self.conhidadj = nn.Linear(28, 128)

	def set_value(self, cond, ma,start):
		self.cond = cond
		self.ma = ma
		self.start = start

	def forward(self, data,timesteps=None, cond=None,samp=False):
		condition_x = cond.x.float()
		condition_pos = cond.pocket_pos
		batch = cond.batch
		noitarget = data

		temb = TimestepEmbedding(timesteps, 128),
		target_hidden = self.mlp2(noitarget)
		temb = temb[0]
		temb = self.Proj(temb)  #
		target_hidden = target_hidden + temb
		target_hidden = self.mlp(target_hidden)
		condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
		condition_hidden = self.conhidadj(condition_hidden)
		condition_dense, mask = to_dense_batch(condition_hidden, batch)

		if samp==True and self.start == True:
			condition_dense = condition_dense.repeat(1000, 1, 1)  # mod./
			mask = mask.repeat(1000, 1)  # mod.
			mask = mask.unsqueeze(1).unsqueeze(2)
			start = False
			self.set_value(condition_dense, mask, start)

		target_cond, attention = self.attention_model(target_hidden, self.cond, self.cond, self.ma)
		target_merged = torch.cat([target_cond.squeeze(1), target_hidden], dim=1)
		output = self.out(target_merged)

		return output


class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.go = nn.Linear(256,256)#
		self.mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(256, 512),
			nn.SiLU(),
			nn.BatchNorm1d(512),
			nn.Linear(512, 1024),
			nn.SiLU(),
			nn.BatchNorm1d(1024),
			nn.Linear(1024, 512),
			nn.SiLU(),
			nn.BatchNorm1d(512),
			nn.Linear(512, 256),
			nn.SiLU(),
			nn.BatchNorm1d(256),
			nn.Linear(256, 128),
			nn.SiLU(),
			nn.BatchNorm1d(128),
			nn.Linear(128, 64),
			nn.SiLU(),
			nn.BatchNorm1d(64),
			nn.Linear(64, 32),
			nn.SiLU(),
			nn.BatchNorm1d(32),
			nn.Linear(32, 16),
			nn.SiLU(),
			nn.BatchNorm1d(16),
			nn.Linear(16, 8),
			nn.BatchNorm1d(8),
			nn.SiLU(),
			nn.Linear(8,2),
		)



	def forward(self, x, timesteps=None):
		bs = len(x)
		temb = TimestepEmbedding(timesteps, 128),
		x = torch.cat([x, temb[0]], dim=1)
		x = self.go(x)
		return self.mlp(x)


class Net_old(torch.nn.Module):
	def __init__(self):
		super(Net_old, self).__init__()
		self.go = nn.Linear(129,256)#
		self.mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(256, 512),
			nn.SiLU(),
			nn.BatchNorm1d(512),
			nn.Linear(512, 1024),
			nn.SiLU(),
			nn.BatchNorm1d(1024),
			nn.Linear(1024, 512),
			nn.SiLU(),
			nn.BatchNorm1d(512),
			nn.Linear(512, 256),
			nn.SiLU(),
			nn.BatchNorm1d(256),
			nn.Linear(256, 128),
			nn.SiLU(),
			nn.BatchNorm1d(128),
			nn.Linear(128, 64),
			nn.SiLU(),
			nn.BatchNorm1d(64),
			nn.Linear(64, 32),
			nn.SiLU(),
			nn.BatchNorm1d(32),
			nn.Linear(32, 16),
			nn.SiLU(),
			nn.BatchNorm1d(16),
			nn.Linear(16, 8),
			nn.BatchNorm1d(8),
			nn.SiLU(),
			nn.Linear(8,2),
		)



	def forward(self, x, timesteps=None):
		bs = len(x)
		if np.prod(timesteps.size()) == 1:

			h_time = torch.empty_like(x[:, 0:1]).fill_(timesteps.item())
		else:
			h_time = timesteps.view(bs, 1).repeat(1, 1)
			h_time = h_time.view(bs, 1)

		x = torch.cat([x, h_time], dim=1)
		x = self.go(x)
		return self.mlp(x)

class Net2(nn.Module):
	def __init__(self):
		super(Net2, self).__init__()
		self.elu = nn.ELU()
		self.fc1 = nn.Linear(512, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.dropout1 = nn.Dropout(0.2)
		self.fc2 = nn.Linear(512, 1024)
		self.bn2 = nn.BatchNorm1d(1024)
		self.dropout2 = nn.Dropout(0.2)
		self.fc3 = nn.Linear(1024, 2048)
		self.bn3 = nn.BatchNorm1d(2048)
		self.dropout3 = nn.Dropout(0.2)
		self.fc4 = nn.Linear(2048, 1024)
		self.bn4 = nn.BatchNorm1d(1024)
		self.dropout4 = nn.Dropout(0.2)
		self.fc5 = nn.Linear(1024, 512)
		self.bn5 = nn.BatchNorm1d(512)
		self.dropout5 = nn.Dropout(0.2)
		self.fc6 = nn.Linear(512, 128)
		self.bn6 = nn.BatchNorm1d(128)
		self.dropout6 = nn.Dropout(0.2)
		self.fc7 = nn.Linear(128, 2)
		self.fct = nn.Linear(129,512)
		self.time_embed = nn.Sequential(
			nn.Linear(128, 256),
			nn.SiLU(),
			nn.Linear(256, 128),
		)

	def forward(self, x,timesteps=None):
		bs = len(x)
		if np.prod(timesteps.size()) == 1:
			h_time = torch.empty_like(x[:, 0:1]).fill_(timesteps.item())
		else:
			h_time = timesteps.view(bs, 1).repeat(1, 1)
			h_time = h_time.view(bs, 1)
		x = torch.cat([x, h_time], dim=1)

		x = self.fct(x)
		x = self.elu(x)
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.elu(x)
		x = self.dropout1(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = self.elu(x)
		x = self.dropout2(x)
		x = self.fc3(x)
		x = self.bn3(x)
		x = self.elu(x)
		x = self.dropout3(x)
		x = self.fc4(x)
		x = self.bn4(x)
		x = self.elu(x)
		x = self.dropout4(x)
		x = self.fc5(x)
		x = self.bn5(x)
		x = self.elu(x)
		x = self.dropout5(x)
		x = self.fc6(x)
		x = self.bn6(x)
		x = self.elu(x)
		x = self.dropout6(x)
		x = self.fc7(x)
		return x

class reg_Net(torch.nn.Module):
	def __init__(self):
		super(reg_Net, self).__init__()
		self.go = nn.Linear(129,256)#
		self.mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(256, 512),
			nn.SiLU(),
			nn.BatchNorm1d(512),
			nn.Linear(512, 1024),
			nn.SiLU(),
			nn.BatchNorm1d(1024),
			nn.Linear(1024, 512),
			nn.SiLU(),
			nn.BatchNorm1d(512),
			nn.Linear(512, 256),
			nn.SiLU(),
			nn.BatchNorm1d(256),
			nn.Linear(256, 128),
			nn.SiLU(),
			nn.BatchNorm1d(128),
			nn.Linear(128, 64),
			nn.SiLU(),
			nn.BatchNorm1d(64),
			nn.Linear(64, 32),
			nn.SiLU(),
			nn.BatchNorm1d(32),
			nn.Linear(32, 16),
			nn.SiLU(),
			nn.BatchNorm1d(16),
			nn.Linear(16, 8),
			nn.BatchNorm1d(8),
			nn.SiLU(),
			nn.Linear(8,1),
		)



	def forward(self, x, timesteps=None):
		bs = len(x)
		if np.prod(timesteps.size()) == 1:
			h_time = torch.empty_like(x[:, 0:1]).fill_(timesteps.item())
		else:
			h_time = timesteps.view(bs, 1).repeat(1, 1)
			h_time = h_time.view(bs, 1)
		x = torch.cat([x, h_time], dim=1)
		x = self.go(x)
		return self.mlp(x)

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
		temb = TimestepEmbedding(timesteps, 128),
		h_time = self.time_embed(temb[0])
		x = torch.cat([x, h_time], dim=1)
		x = self.go(x)
		return self.mlp(x)

class reg_Net_newt(torch.nn.Module):
	def __init__(self):
		super(reg_Net_newt, self).__init__()
		self.go = nn.Linear(256,256)#
		self.mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(256, 512),
			nn.SiLU(),
			nn.BatchNorm1d(512),
			nn.Linear(512, 1024),
			nn.SiLU(),
			nn.BatchNorm1d(1024),
			nn.Linear(1024, 512),
			nn.SiLU(),
			nn.BatchNorm1d(512),
			nn.Linear(512, 256),
			nn.SiLU(),
			nn.BatchNorm1d(256),
			nn.Linear(256, 128),
			nn.SiLU(),
			nn.BatchNorm1d(128),
			nn.Linear(128, 64),
			nn.SiLU(),
			nn.BatchNorm1d(64),
			nn.Linear(64, 32),
			nn.SiLU(),
			nn.BatchNorm1d(32),
			nn.Linear(32, 16),
			nn.SiLU(),
			nn.BatchNorm1d(16),
			nn.Linear(16, 8),
			nn.BatchNorm1d(8),
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
		temb = TimestepEmbedding(timesteps, 128),
		h_time = self.time_embed(temb[0])
		x = torch.cat([x, h_time], dim=1)
		x = self.go(x)
		return self.mlp(x)

	def predict_qed(self,x,t):
		return self.forward(x,t)


class deeper_cls_net(nn.Module):
	def __init__(self, condition_time=True):
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
								 nn.SiLU(),
								 nn.Linear(128, 96),
								 nn.SiLU(),
								 nn.Linear(96, 64),
								 nn.SiLU(),
								 nn.Linear(64, 32),
								 nn.SiLU(),
								 nn.Linear(32, 16),
								 nn.SiLU(),
								 nn.Linear(16, 8),
								 nn.SiLU(),
								 nn.Linear(8, 4),
								 nn.SiLU(),
								 nn.Linear(4, 2)
								 )
	def forward(self,noidata, timesteps=None, batch=None):
		noitarget = noidata
		transed_target = self.allfirst(noitarget)
		temb = TimestepEmbedding(timesteps,128),
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
		return output