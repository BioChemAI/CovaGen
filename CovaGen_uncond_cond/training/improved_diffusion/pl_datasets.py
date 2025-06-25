'''
Since the Dataset Class we use is defined as Crossdockdataset in optdiffusion/crossdock_dataset.py, here in this script,
we only implement the methods to load the dataset.
'''

from PIL import Image
import blobfile as bf
import pickle
from mpi4py import MPI
import numpy as np
import torch

from optdiffusion import crossdock_dataset,esm_dataset
from torch_geometric.data import DataLoader
from torch.utils.data import Subset, Dataset
from torch.nn.utils.rnn import pad_sequence



def load_data_smi( * , batch_size, vae_dir = None, dataset_save_path = None, data_dir=None, class_cond=False, deterministic=False, data_mood = "train"):
    # 用了自定义的split方法，对CrossDock数据集管用
    datadir = data_dir
    print(f"now {data_mood}")
    dataset = crossdock_dataset.PocketLigandPairDataset(
        f'{datadir}',  # 这是 raw path
        vae_path=f'{vae_dir}',
        save_path=f'{dataset_save_path}'
    )
    print("vae is :",vae_dir)
    datafull, subsets = split(dataset,'/workspace/datas/11/split_by_name.pt')
    #if deterministic:
    #    loader = DataLoader(
    #        dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #    )
    #else:
    train, val = subsets['train'], subsets['test']
    print(f"Number of Training Data:{len(train)}")
    print(f"Number of Test Data:{len(val)}")
    follow_batch = ['protein_pos', 'ligand_pos'] # 还是有必要去再看看follow_batch的用法的。这个应该是pyg的Dataloader的？

    if data_mood =="train":
        loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, follow_batch=follow_batch
        )
        while True:
            yield from loader
    elif data_mood =="sample":
        loader = DataLoader(
            val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, follow_batch=follow_batch
        )
        print("now sample,num:",len(loader))
        # return loader # 这个warning具体什么含义呢？
        while True:
            yield from loader


def split(dataset,split_file):
    split_by_name = torch.load(split_file)
    split = {
        k: [dataset.name2id[n] for n in names if n in dataset.name2id]
        for k, names in split_by_name.items()
    }
    subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
    return dataset, subsets

def load_data2(
    *, data_dir, batch_size,class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = SimpleDataset(
        data_path=data_dir
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader

def smiles_to_one_hot(smiles_string):
    max_length = 128
    smiles_dict = {
        0: 'C', 1: 'N', 2: '(', 3: '=', 4: 'O', 5: ')', 6: 'c', 7: '1', 8: '2', 9: '#',
        10: 'n', 11: 'Cl', 12: '-', 13: '3', 14: 'o', 15: 'Br', 16: '[', 17: '@', 18: 'H',
        19: ']', 20: 's', 21: '4', 22: 'B', 23: 'F', 24: 'S', 25: '5', 26: 'I', 27: '6',
        28: '/', 29: 'i', 30: '+', 31: '\\', 32: 'P', 33: '7', 34: 'Z', 35: 'r', 36: 'M',
        37: 'g', 38: 'L', 39: 'f', 40: 'T', 41: 'e', 42: 'K', 43: 'V', 44: 'A', 45: 'l',
        46: 'b', 47: '8', 48: '9', 49: 'a', 50: 't', 51: 'Y', 52: 'G', 53: 'R', 54: 'u',
        55: 'p', 56: 'h', 57: 'U', 58: 'd', 59: 'W', 60: '%', 61: '0', 62: 'X', 63: '_',
        64: '<end>'
    }
    num_chars = len(smiles_dict)
    one_hot_vector = np.zeros((max_length, num_chars))
    for i, char in enumerate(smiles_string):
        if i >= max_length:
            break
        if char in smiles_dict.values():
            char_index = [k for k, v in smiles_dict.items() if v == char][0]
            one_hot_vector[i, char_index] = 1
    return one_hot_vector

def generate_mask(smiles_string, max_length):
    mask = [1] * len(smiles_string)
    if len(smiles_string) < max_length:
        mask += [0] * (max_length - len(smiles_string))
    return mask

class SimpleDataset(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data= []
        with open("/workspace/codes/vaemodel/zinc_128merged_withcrossdock.txt", 'r') as f:
            for line in f:
                self.data.append(line.rstrip())  #
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        msk = generate_mask(self.data[idx],128)
        return torch.Tensor(smiles_to_one_hot(self.data[idx])),msk

def load_data_esm( * , batch_size, vae_dir = None, dataset_save_path = None, data_dir=None, class_cond=False, deterministic=False, data_mood = "train"):
    datadir = data_dir
    print(f"now {data_mood}")
    dataset = esm_dataset.SequenceLigandPairDataset(
        f'{datadir}',
        vae_path=f'{vae_dir}',
        save_path=f'{dataset_save_path}'
    )
    print("vae is :",vae_dir)
    print("just training data")

    if data_mood =="train":
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
        )
        while True:
            yield from loader
    elif data_mood =="sample":
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True,
        )
        print("now sample,num:",len(loader))
        # return loader # 这个warning具体什么含义呢？
        while True:
            yield from loader

def collate_fn(batch):
    # 获取每个样本的原始长度
    original_lengths = [len(seq) for seq in batch]

    # 找到batch中最长的长度
    max_length = max(original_lengths)

    # 对每个样本进行padding，使其长度一致
    padded_batch = [torch.nn.functional.pad(torch.tensor(seq), (0, max_length - len(seq))) for seq in batch]

    # 创建masking tensor，标记padding的位置
    mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    for i, length in enumerate(original_lengths):
        mask[i, :length] = 1

    return pad_sequence(padded_batch, batch_first=True), mask