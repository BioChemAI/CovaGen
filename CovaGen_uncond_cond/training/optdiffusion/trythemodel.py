import torch
from torch import nn
from EGNNEncoder_pyg import EGNNEncoder
from MultiHeadAttentionLayer import MultiHeadAttentionLayer
import numpy as np
from torch_geometric.utils import to_dense_batch
class Dynamics(nn.Module):  # 首先明确这个部分对应了ldm模型中的哪个位置，这样才好嵌入
    def __init__(self,condition_dim,target_dim,hid_dim,condition_layer,
                 n_heads, dropout=0.2, condition_cutoff=5,condition_time=True,device=torch.device('cuda:0')):
        super().__init__()
        self.mlp = nn.Linear(target_dim+1 if condition_time else target_dim,hid_dim) # 一个mlp层 是一个linear，此处128->64dim
        self.condition_model = EGNNEncoder(condition_dim, hid_dim, layers=condition_layer, cutoff=condition_cutoff)
        # EGNNEncoder这里是对pocket进行编码嵌入
        self.attention_model = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.condition_time = condition_time    # 这个time是干嘛的？
        self.out = nn.Linear(hid_dim,target_dim)    # 再过一个linear 64-128dim输出。

    def forward(self,data,t=None):
        condition_x = data.x.float()    # 注意下这个dottable在哪儿实现的捏？
        condition_pos = data.pocket_pos
        batch = data.batch  # 回忆，dataloader中一个batch的batch属性存储的是什么？
        target = data.target
        num_nodes = data.nodes
        bs = max(batch)+1
        target = target.view(bs,-1)

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(target[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, 1)
                h_time = h_time.view(bs , 1)
            target_with_time = torch.cat([target, h_time], dim=1)
            target_hidden = self.mlp(target_with_time)
        else:
            target_hidden = self.mlp(target)    # 过mlp

        coors,condition_hidden = self.condition_model(condition_pos, condition_x,batch=batch)
        condition_dense,mask = to_dense_batch(condition_hidden,batch)
        mask = mask.unsqueeze(1).unsqueeze(2)

        target_merged,attention = self.attention_model(target_hidden,condition_dense,condition_dense,mask)
        output = self.out(target_merged)
        error = target-output
        return error

if __name__=='__main__':
    from torch_geometric.data import DataLoader
    from crossdock_dataset import PocketLigandPairDataset
    device = torch.device('cuda:0')
    dataset = PocketLigandPairDataset('/workspace/dataset/lpy/sbdd_with_protein/',
                                      vae_path='../download_model/trans4x-256_zinc.ckpt',
                                      save_path='/workspace/dataset/lpy/tmp/ldm_dataset/')
    loader = DataLoader(dataset,batch_size=2)

    model = Dynamics(condition_dim=28,target_dim=128,hid_dim=64,condition_layer=3,n_heads=2,condition_time=False).to(device)
    for batch in loader:
        batch = batch.to(device)
        print(batch)
        out = model(batch, t=torch.tensor(1))
        print(out)
