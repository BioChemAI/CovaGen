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

# class Dynamics(nn.Module):  # ������ȷ������ֶ�Ӧ��ldmģ���е��ĸ�λ�ã������ź�Ƕ��
#     def __init__(self,condition_dim,target_dim,hid_dim,condition_layer,
#                  n_heads, dropout=0.2, condition_cutoff=5,condition_time=False,device=torch.device('cuda:0')):
#         super().__init__()
#         #self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # һ��mlp�� ��һ��linear���˴�128->64dim
#         self.mlp = nn.Linear(target_dim,hid_dim)
#         self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
#         # EGNNEncoder�����Ƕ�pocket���б���Ƕ��
#         self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
#         #self.condition_time = condition_time    # ���time�Ǹ���ģ�
#         self.out = nn.Linear(hid_dim,target_dim)    # �ٹ�һ��linear 64-128dim�����
#
#     def forward(self,data,noidata,t=None, samp = False):
# # TODO: forward(self, condion_x,condition_pos, noise_target,batch,t=None, samp = False)
#         condition_x = data.x.float()    # ע�������dottable���Ķ�ʵ�ֵ���������Щ�����ѵ���torch_geom���dataset����Щ���ԣ�
#         condition_pos = data.pocket_pos
#         batch = data.batch  # ���䣬dataloader��һ��batch��batch���Դ洢����ʲô��
#         target = data.target # ���targetӦ�ü�������
#         noitarget = noidata
#
#         print("samp shape of noi:",noidata.shape)
#
#         if samp:
#             noitarget = noidata.squeeze(0)
#
#         print("noised length:",len(noitarget))
#         print("is noised allright?:", noitarget)
#
#         num_nodes = data.nodes
#         bs = max(batch)+1
#         target = target.view(bs,-1)
#         #noitarget = noitarget.view(bs,-1)
#         print("viewed noitarget:", len(noitarget))
#         #if self.condition_time:
#         #    if np.prod(t.size()) == 1:
#         #        # t is the same for all elements in batch.
#         #        h_time = torch.empty_like(target[:, 0:1]).fill_(t.item())
#         #    else:
#         #        h_time = t.view(bs, 1).repeat(1, 1)
#         #        h_time = h_time.view(bs , 1)
#         #    target_with_time = torch.cat([target, h_time], dim=1)
#         #    target_hidden = self.mlp(target_with_time)
#         #else:
#         target_hidden = self.mlp(noitarget) # TODO:input error
#         print(len(target_hidden))
#
#         coors,condition_hidden = self.condition_model(condition_pos, condition_x,batch=batch)
#         condition_dense,mask = to_dense_batch(condition_hidden,batch)
#         print(condition_dense.shape)
#         print(mask.shape)
#         mask = mask.unsqueeze(1).unsqueeze(2)
#         print("unsqed mask:", mask.shape)
#         target_merged,attention = self.attention_model(target_hidden,condition_dense,condition_dense,mask)
#         output = self.out(target_merged)
#         # �����������δ�޸ĵ�target��batchsize*ldim����״����out��batchsize*1*ldim����״�����¼�����һ�������״�Ķ���
#         #�м��Ǻ�ӵ�һ��
#         target = target.unsqueeze(1)
#         #1
#         error = target-output
#         error1 = error.squeeze(1)
#         if samp:
#             return output
#         elif not samp:
#             return error1

# Now imma do some change to the model...
class Dynamics_samp(nn.Module):  # ������ȷ������ֶ�Ӧ��ldmģ���е��ĸ�λ�ã������ź�Ƕ��
    def __init__(self, condition_dim, target_dim, hid_dim, condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5, condition_time=True, device=torch.device('cuda:0'), sampling = False):
        super().__init__()
        # self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # һ��mlp�� ��һ��linear���˴�128->64dim
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim)
        #self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        self.condition_model = SchNetEncoder(hidden_channels=28, cutoff=condition_cutoff)
        # EGNNEncoder�����Ƕ�pocket���б���Ƕ��
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time    # ���time�Ǹ���ģ�
        self.out = nn.Linear(hid_dim, target_dim)  # �ٹ�һ��linear 64-128dim�����
        self.conhidadj = nn.Linear(28, 64)
        
    def forward(self, data, condition_x,condition_pos, noidata, batch, t=None, samp = True, look=1):
        print(samp)
        # TODO: forward(self, condition_x,condition_pos, noise_target,batch,t=None, samp = False)
        condition_x = condition_x  # ע�������dottable���Ķ�ʵ�ֵ���������Щ�����ѵ���torch_geom���dataset����Щ���ԣ�
        condition_pos = condition_pos

        batch = batch  # ���䣬dataloader��һ��batch��batch���Դ洢����ʲô��
        target = data.target  # ��Ϊԭtarget��
        noitarget = noidata

        #print("samp shape of noi:", noidata.shape)

        # if samp:    # This needs some alters...
        #     noitarget = noidata.squeeze(0)

        #print("noised length:", len(noitarget))
        #print("is noised allright?:", noitarget)
        print("now look",look)
        num_nodes = data.nodes
        bs = max(batch) + 1
        target = target.view(bs, -1)
        # noitarget = noitarget.view(bs,-1)
        #print("viewed noitarget:", len(noitarget))
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

        condition_hidden = self.condition_model(condition_x, condition_pos, batch=batch)
        condition_hidden = self.conhidadj(condition_hidden)
        condition_dense, mask = to_dense_batch(condition_hidden, batch)
        if look == 0:
            condition_dense=condition_dense.repeat(6,1,1)
            target_hidden=target_hidden.repeat(6,1)
        print(condition_dense.shape)
        print("shape de tar_hidden",target_hidden.shape)
        #print(mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(2)
        #print("unsqed mask:", mask.shape)
        target_merged, attention = self.attention_model(target_hidden, condition_dense, condition_dense, mask)
        output = self.out(target_merged)
        # �����������δ�޸ĵ�target��batchsize*ldim����״����out��batchsize*1*ldim����״�����¼�����һ�������״�Ķ���
        # �м��Ǻ�ӵ�һ��
        target = target.unsqueeze(1)
        # 1
        error = target - output
        error1 = error.squeeze(1)
        output1 = output.squeeze(1)
        if samp:
            print(nowwesample)
            return output1
        elif not samp:
            return error1

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