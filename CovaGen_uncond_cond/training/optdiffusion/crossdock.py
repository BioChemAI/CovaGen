"""
This script is for the data generation of the transvae, didn't change much of the original code.
"""

import os,sys
sys.path.append(os.path.dirname(sys.path[0]))
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from torch import utils
from tqdm.auto import tqdm
import torch.nn.functional as F
from optdiffusion.protein_ligand_process import PDBProtein, smiles_to_embed
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
sys.path.append("../")
from transvae.trans_models import TransVAE
from scipy.spatial.transform import Rotation
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))
class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, vae_path, save_path, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'crossdocked_pocket10/', 'index.pkl')
        self.processed_path = os.path.join(save_path,
                                          'crossdocked_pocket10_processed_forsmiles1.lmdb') # 这个就是lmdb那里那个db连接到的数据的位置！
        self.name2id_path = os.path.join(save_path,
                                         'crossdocked_pocket10_name2id.pt')
        self.transform = transform
        self.db = None
        self.keys = None
        self.vae_path = vae_path

        if not os.path.exists(self.processed_path):
            self._process()
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

#   LMDB是非常快的内存映射型数据库，LMDB使用内存映射文件，可以提供更好的输入/输出性能，对于用于神经网络的大型数据集( 比如 ImageNet )，可以将其存储在 LMDB 中。
#   LMDB提供 key-value 存储，其中每个键值对都是我们数据集中的一个样本。LMDB的主要作用是提供数据管理，可以将各种各样的原始数据转换为统一的key-value存储。
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None


    def _process(self):

        with open(self.index_path, 'rb') as f: # indexpath：raw_path, 'crossdocked_pocket10/', 'index.pkl'
            index = pickle.load(f)
        #index = index[:1000] #for dev
        ### convert to smiles; remove duplicate
        no_pocket=0
        none_mol=0
        success=0
        source_list = []
        pbar = tqdm(index)
        processed_smi = []
        for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(pbar): # 所以说这个index文件应该是pocket_fn, ligand_fn, _, rmsd_str的路径记录的结构。（去下下来看一下？）
            sdf_path = os.path.join(self.raw_path, 'crossdocked_pocket10/', ligand_fn)
            mol = next(iter(Chem.SDMolSupplier(sdf_path)))# SDMolSupplier是一个读sdf文件的方法，该方式读取的文件存成的是一个列表的形式，支持列表的操作。
            if mol is None:
                none_mol+= 1 # 没分子。。
                continue
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)    # 转之，mol转为smi
            processed_smi.append(smiles)
            source_list.append([pocket_fn,ligand_fn,smiles])    # 创建了一个存储 pocketfn，ligandfn，smi的列表
            success+=1  # 成功一个
            pbar.set_postfix({'no_pocket':no_pocket,'none_mol':none_mol,'success':success})
        print(len(processed_smi))
        with open("/workspace/stu/ltx/nt/dataset/newsmis.txt", "w") as f2:
            for dt in processed_smi:
                f2.write(dt+'\n')

        pocket_smi_set = set()  # 创建pocket smi的空集合？
        source_list_new = []
        for source_item in source_list:
            ps_tuple = (tuple(source_item[0]), tuple(source_item[2]))   # tuple之tuple。 是（（pocketfn），（smiles））的格式
            if ps_tuple not in pocket_smi_set:
                pocket_smi_set.add(ps_tuple)    # set增加元素用add！别再忘了！
                source_list_new.append(source_item) # 酱紫去重。重复的情况是什么呢？是已经有了pocket对应ligand的关系了
        print('{} samples, after remove duplicate, {} left'.format(len(source_list),len(source_list_new)))

        # featurize
        vae = TransVAE(load_fn=self.vae_path)   # 之前train的vae
        processed_data=[]
        fail=0
        success=0
        pbar = tqdm(source_list_new)

        for i,(pocket_fn, ligand_fn, smiles) in enumerate(pbar):
            smiles_emb = smiles_to_embed(smiles,vae_model=vae)  # smiles emb用的就是vae输出的mu。
            if smiles_emb is None:
                fail+=1
                continue    # 那估计是字典不够大了。。
            pocket_dict = PDBProtein(os.path.join(self.raw_path, 'crossdocked_pocket10/', pocket_fn)).to_dict_atom()
            # 先大致记住这是一个把蛋白质序列转一个表示的方法吧，这里pocket_dict记录了一个pocket的一系列信息。
            data = {'pocket': pocket_dict, 'smiles_emb': smiles_emb, 'smiles': smiles,
                    'protein_filename': pocket_fn, 'ligand_filename': ligand_fn
                    } # 最后获得的data的是一个dict，重点的是其中的pocket和smiles_emb。我的理解是一个pocket可以有不同种类的ligand和其结合，这样可以对pocket取一个dict来形成这个
            # 一对多的关系。
            processed_data.append(data)
            success+=1
            pbar.set_postfix(dict(success=success,fail=fail))

        # save the data
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with db.begin(write=True, buffers=True) as txn:
            for idx,data in enumerate(processed_data):
                txn.put(
                    key=str(idx).encode(),
                    value=pickle.dumps(data)
                )
        db.close()  # 最后采用了lmdb的方法保存的。应该是在这个路径：crossdocked_pocket10_processed.lmdb

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                #data = self.__getitem__(i)
                data = self.getdata(i)
            except AssertionError as e:
                print(i, e)
                continue
            if data['protein_filename']:
                name = (data['protein_filename'], data['ligand_filename'])
                name2id[name] = i
                print(f"{i} is good")
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def getdata(self,idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        return data


    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx] # 返回的key[idx]
        data = pickle.loads(self.db.begin().get(key)) # 这一步就是从lmdb中取数据 怎么读的数据和我想象的不一样呢
        # return data
        ### onehot pocket
        pocket = data['pocket']
        element = F.one_hot(pocket['atom'], num_classes=6)  # ['C', 'N', 'H', 'S', 'O']
        amino_acid = F.one_hot(pocket['res'], num_classes=21)
        is_backbone = pocket['is_backbone'].view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        num_nodes = torch.LongTensor([len(x)])
        pygdata = Data(x=x, pocket_pos=pocket['pos'],nodes=num_nodes,target=data['smiles_emb'],id=idx) # id是idx，Data中id是一个列表，其中有batchsize的数目的idx，用于getitem。
        # target是batchsize*len(idx)的，也就是有batchsize个smiemb！node也是batchsize个

        if self.transform is not None:
            pygdata = self.transform(pygdata)
        return pygdata

def random_rotation_translation(translation_distance):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1,3)
    t = t/np.sqrt(np.sum(t*t))
    length = np.random.uniform(low=0,high=translation_distance)
    t = t*length
    return torch.from_numpy(rotation_matrix.astype(np.float32)),torch.from_numpy(t.astype(np.float32))

class Rotate_translate_Transforms(object):
    def __init__(self, distance):
        self.distance = distance
    def __call__(self, data):
        R,t = random_rotation_translation(self.distance)
        pos = data.protein_pos
        new_pos = (R@pos.T).T+t
        data.pocket_pos = new_pos
        return data







if __name__ == '__main__':
    from torch_geometric.transforms import Compose
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str)
    # args = parser.parse_args()
    from torch_geometric.data import DataLoader
    from torch.utils.data import random_split
    from torch.utils.data import Subset
    import sys
    sys.path.append('..')
    import os


    # transform = Compose([FeaturizeProtein(),FeaturizeLigand()])
    device = torch.device('cuda:0')

    dataset = PocketLigandPairDataset('/workspace/stu/ltx/nt/dataset/dataset/', # 这是 raw path
                                      vae_path='/workspace/stu/ltx/nt/045_trans1x-128_zinc.ckpt',
                                      save_path='/workspace/stu/ltx/nt/dataset/dataset/processed/')
    loader = DataLoader(dataset, batch_size=1)

    # train,valid = random_split(dataset,lengths=[int(len(dataset)*0.9),int(len(dataset)-int(len(dataset)*0.9))])
    # data = train[0]
    # def split(dataset,split_file):
    #     split_by_name = torch.load(split_file)
    #     split = {
    #         k: [dataset.name2id[n] for n in names if n in dataset.name2id]
    #         for k, names in split_by_name.items()
    #     }
    #     subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
    #     return dataset, subsets
    # dataset,subsets = split(dataset,'/workspace/stu/ltx/nt/dataset/dataset/crossdocked_pocket10/split_by_name.pt')
    # train, val = subsets['train'], subsets['test']
    # print(len(dataset),len(train),len(val))
    #
    # follow_batch =  ['protein_pos','ligand_pos']
    # j = 0
    # loader = DataLoader(val,batch_size=100,follow_batch=follow_batch)
    # for batch in loader:
    #     print(batch)
    #     print('done')
    #     j +=1
    # print(j)