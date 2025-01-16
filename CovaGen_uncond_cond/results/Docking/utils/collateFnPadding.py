import torch
from torch.nn.utils.rnn import pad_sequence

class pad_batch:
    def __init__(self, batch, src_pad_idx, trg_pad_idx):
        self.src = pad_sequence([torch.tensor(t[0]) for t in batch], padding_value=src_pad_idx, batch_first=True)
        self.trg = pad_sequence([torch.tensor(t[1]) for t in batch], padding_value=trg_pad_idx, batch_first=True)

class collateFnPadding:
    def __init__(self, src_pad_idx, trg_pad_idx):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def __call__(self, batch) -> pad_batch:
        return pad_batch(batch, self.src_pad_idx, self.trg_pad_idx)

# class pad_batch_with_segment:
#     def __init__(self, batch, src_pad_idx, trg_pad_idx):
#         self.src = pad_sequence([torch.tensor(t[0]) for t in batch], padding_value=src_pad_idx, batch_first=True)
#         self.trg = pad_sequence([torch.tensor(t[1]) for t in batch], padding_value=trg_pad_idx, batch_first=True)
#         self.seg = pad_sequence([torch.tensor(t[2]) for t in batch], padding_value=-1, batch_first=True) + 1

# class collateFnPaddingWithSegment:
#     def __init__(self, src_pad_idx, trg_pad_idx):
#         self.src_pad_idx = src_pad_idx
#         self.trg_pad_idx = trg_pad_idx

#     def __call__(self, batch) -> pad_batch_with_segment:
#         return pad_batch_with_segment(batch, self.src_pad_idx, self.trg_pad_idx)
