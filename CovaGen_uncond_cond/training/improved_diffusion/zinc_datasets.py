from torch.utils.data import DataLoader, Dataset
import blobfile as bf
import pickle
from mpi4py import MPI
import numpy as np
import torch

class Zinc_dataset(Dataset):
	def __init__(self, file):
		super().__init__()
		# self.data = pd.read_csv(csv_file,dtype=np.float32)
		# self.labels = self.data["type"].values
		# self.data = self.data.drop("type", axis=1).values
		with open(file, "rb") as f:
			self.data = pickle.load(f)
			print("loaded")


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		return sample

def load_zinc_data(
	*,
	data_dir,
	batch_size,
	reg=False,
	class_cond=False,
	deterministic=False,
	random_crop=False,
	random_flip=True,
	state='train'
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
	:param random_crop: if True, randomly crop the images for augmentation.
	:param random_flip: if True, randomly flip the images for augmentation.
	"""
	if not data_dir:
		raise ValueError("unspecified data directory")
	file_path = data_dir
	datas = Zinc_dataset(file_path)
	train_loader = DataLoader(
		datas, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
	)
	while True:
		yield from train_loader