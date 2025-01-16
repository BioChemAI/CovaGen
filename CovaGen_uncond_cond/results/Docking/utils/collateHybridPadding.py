import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch

#batch the pyg data object and torch tensor object

class pad_batch:
    def __init__(self, batch, trg_pad_idx,follow_batch=None,exclude_keys=None):
        self.src = Batch.from_data_list([t[0] for t in batch], follow_batch,exclude_keys)
        self.trg = pad_sequence([torch.tensor(t[1]) for t in batch], padding_value=trg_pad_idx, batch_first=True)

class collateHybridPadding:
    def __init__(self, trg_pad_idx, follow_batch=None,exclude_keys=None):

        self.trg_pad_idx = trg_pad_idx
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys


    def __call__(self, batch) -> pad_batch:
        return pad_batch(batch, self.trg_pad_idx, self.follow_batch, self.exclude_keys)
