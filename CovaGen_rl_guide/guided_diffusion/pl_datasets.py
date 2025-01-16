'''
Here in this script is the methods to load the process CrossDock dataset. This is based on image_dataset.py in the implementation of IDDPM
'''

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from optdiffusion import crossdock_dataset,esm_dataset
from torch_geometric.data import DataLoader
from torch.utils.data import Subset

def load_data_smi( * , batch_size, vae_dir = None, dataset_save_path = None, data_dir=None, class_cond=False, deterministic=False, data_mood = "train"):
    datadir = data_dir
    print(f"now {data_mood}")
    dataset = crossdock_dataset.PocketLigandPairDataset(
        f'{datadir}',
        vae_path=f'{vae_dir}',
        save_path=f'{dataset_save_path}'
    )
    print("vae is :",vae_dir)
    datafull, subsets = split(dataset,'../Models/split_by_name.pt')
    train, val = subsets['train'], subsets['test']
    follow_batch = ['protein_pos', 'ligand_pos']
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
        while True:
            yield from loader

def load_data_esm( * , batch_size, vae_dir = None, dataset_save_path = None, data_dir=None, class_cond=False, deterministic=False, data_state = "train",sequence=None):
    datadir = data_dir

    if sequence is not None:
        dataset = esm_dataset.SequenceLigandPairDataset(
            f'{datadir}',
            vae_path=f'{vae_dir}',
            save_path=f'{dataset_save_path}',
            sequence = sequence,
        )
        print("vae is :",vae_dir)
        print("just training data")
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True,
        )
        while True:
            yield from loader

    else:
        dataset = esm_dataset.SequenceLigandPairDataset(
            f'{datadir}',
            vae_path=f'{vae_dir}',
            save_path=f'{dataset_save_path}'
        )
        if data_state =="train":
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
            )
            while True:
                yield from loader
        elif data_state =="sample":
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
            )
            print("now sample,num:",len(loader))
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