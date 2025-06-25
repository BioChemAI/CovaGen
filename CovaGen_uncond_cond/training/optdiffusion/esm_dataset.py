import os, sys
import random

import pandas as pd

sys.path.append(os.path.dirname(sys.path[0]))
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from torch import utils
from tqdm.auto import tqdm
import time
import torch.nn.functional as F
from optdiffusion.protein_ligand_process import PDBProtein, smiles_to_embed
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
import selfies as sf
sys.path.append("../")
from transvae.trans_models import TransVAE
from transvae.rnn_models import RNNAttn
from scipy.spatial.transform import Rotation
import esm


sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))


def random_rotation_translation(translation_distance):
	rotation = Rotation.random(num=1)
	rotation_matrix = rotation.as_matrix().squeeze()

	t = np.random.randn(1, 3)
	t = t / np.sqrt(np.sum(t * t))##
	length = np.random.uniform(low=0, high=translation_distance)#
	t = t * length
	return torch.from_numpy(rotation_matrix.astype(np.float32)), torch.from_numpy(t.astype(np.float32))


class Rotate_translate_Transforms(object):
	def __init__(self, distance):
		self.distance = distance

	def __call__(self, data):
		R, t = random_rotation_translation(self.distance)
		pos = data.protein_pos
		new_pos = (R @ pos.T).T + t
		data.pocket_pos = new_pos
		return data


class SequenceLigandPairDataset(Dataset):

	def __init__(self, raw_path, vae_path, save_path, transform=None):
		super().__init__()#
		self.csv_path = "/workspace/datas/cross_docked_seqs_train.csv"

		self.raw_path = raw_path.rstrip('/')  ###
		self.index_path = os.path.join(self.raw_path, 'crossdocked_pocket10/', 'index.pkl')
		self.data_path = 'crossdocked_pocket10_processed_esm_preencoded_klhigher80_smi_test_2.lmdb'#
		# self.data_path = 'crossdocked_pocket10_processed_esm_preencoded_klhigher80_smi_full_FULL15_tryall.lmdb'# try
		# self.data_path = '3vlp_2.lmdb'
		print("Data is :", self.data_path)
		self.processed_path = os.path.join(save_path,self.data_path)  # 这个就是lmdb那里那个db连接到的数据的位置！
		self.transform = transform
		self.db = None
		self.keys = None
		self.vae_path = vae_path
#
		if not os.path.exists(self.processed_path):
			self._process()
		# if not os.path.exists(self.name2id_path):
		# 	self._precompute_name2id()

		# self.name2id = torch.load(self.name2id_path)

	#   LMDB是非常快的内存映射型数据库，LMDB使用内存映射文件，可以提供更好的输入/输出性能，对于用于神经网络的大型数据集( 比如 ImageNet )，可以将其存储在 LMDB 中。
	#   LMDB提供 key-value 存储，其中每个键值对都是我们数据集中的一个样本。LMDB的主要作用是提供数据管理，可以将各种各样的原始数据转换为统一的key-value存储。
	def _connect_db(self):
		"""
			Establish read-only database connection
		"""
		assert self.db is None, 'A connection has already been opened.'
		self.db = lmdb.open(
			self.processed_path,
			map_size=41 * (1024 * 1024 * 1024),  # 10GB
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
		fail = 0
		success = 0
		smi_fail = 0
		processed_data = []
		esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
		batch_converter = esm_alphabet.get_batch_converter()
		esm_model.eval()

		vae = RNNAttn(load_fn=self.vae_path)
		df = pd.read_csv(self.csv_path)  # 读取CSV文件
		seqls = df['pocket sequence'].to_list()  # 获取序列列表
		smils = df['smiles'].to_list()  # 获取SMILES列表

		pbar = tqdm(range(len(seqls)))  # 进度条

		# 打开LMDB进行增量写入
		db = lmdb.open(
			self.processed_path,
			map_size=41 * (1024 * 1024 * 1024),  # 10GB
			create=True,
			subdir=False,
			readonly=False,  # Writable
		)

		try:
			cnt = 0  # 计数器
			for i in pbar:
				# 每处理 1000 条数据，进行一次增量写入
				if cnt % 1000 == 0 and cnt > 0:
					print(f"Writing batch to LMDB, processed {cnt} entries so far...")
					with db.begin(write=True) as txn:
						for idx, data in enumerate(processed_data):
							txn.put(
								key=str(cnt + idx).encode(),
								value=pickle.dumps(data)
							)
					processed_data = []  # 清空已写入的数据，释放内存

				indd = i
				value_nodup = [smils[indd]]
				key1 = [('protein1', seqls[indd])]  # 序列信息

				batch_labels, batch_strs, batch_tokens = batch_converter(key1)
				batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)

				with torch.no_grad():
					results = esm_model(batch_tokens, repr_layers=[6], return_contacts=True)
				token_representations = results["representations"][6]  # 获取表征

				for smi in value_nodup:
					try:
						smiles_emb = smiles_to_embed(smi, vae_model=vae)
					except Exception as e:
						fail += 1
						print("Occurred:", e)
						print(f"Success: {success}, Fail: {fail}")
						continue  # 处理失败时跳过

					if smiles_emb is None:
						fail += 1
						smi_fail += 1
						print("Encode failed:", smi_fail)
						continue

					data = {'smiles_emb': smiles_emb, 'smiles': smi, 'token_rep': token_representations}
					processed_data.append(data)
					success += 1
					cnt += 1

					print(f"Success: {success}, Fail: {fail}, Count: {cnt}")

			# 写入剩余的数据
			if processed_data:
				with db.begin(write=True) as txn:
					for idx, data in enumerate(processed_data):
						txn.put(
							key=str(cnt + idx).encode(),
							value=pickle.dumps(data)
						)

			print("Data saved successfully!")

		except Exception as e:
			print(f"Error during LMDB operation: {e}")

		finally:
			db.close()  # 关闭数据库连接


	def _precompute_name2id(self):
		name2id = {}
		for i in tqdm(range(self.__len__()), 'Indexing'):
			try:
				# data = self.__getitem__(i)
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

	def getdata(self, idx):
		if self.db is None:
			self._connect_db()
		key = self.keys[idx]
		data = pickle.loads(self.db.begin().get(key))
		return data

	def __getitem__(self, idx):
		if self.db is None:
			self._connect_db()
		key = self.keys[idx]  # 返回的key[idx]
		data = pickle.loads(self.db.begin().get(key))  # 这一步就是从lmdb中取数据 怎么读的数据和我想象的不一样呢
		# return data
		### onehot pocket
		sample  =data['token_rep'].squeeze(0)
		padded_sample = torch.nn.functional.pad(torch.tensor(sample), (0, 0, 0, 640 - len(sample)))

		# 创建masking tensor，标记padding的位置
		mask = torch.zeros(640, dtype=torch.bool)
		mask[:len(sample)] = 1



		# retdata = (data['token_rep'].squeeze(0),data['seq'],data['smiles_emb'])
		# pygdata = Data(x=data['token_rep'], pocket_pos=pocket['pos'], nodes=num_nodes, target=data['smiles_emb'],
		# 			   id=idx)  # id是idx，Data中id是一个列表，其中有batchsize的数目的idx，用于getitem。
		# target是batchsize*len(idx)的，也就是有batchsize个smiemb！node也是batchsize个

		# if self.transform is not None:
		# 	pygdata = self.transform(pygdata)
		return padded_sample, mask, data['smiles_emb']

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
	split_by_name = torch.load('/workspace/datas/crossdock_protein/crossdocked_pocket10/split_by_name.pt')
	# dataset = PocketLigandPairDataset('/workspace/stu/ltx/nt/dataset/dataset/', # 这是 raw path
	#                                  vae_path='/workspace/stu/ltx/nt/045_trans1x-128_zinc.ckpt',
	#                                  save_path='/workspace/stu/ltx/nt/dataset/dataset/processed/')
	dataset = SequenceLigandPairDataset('/workspace/datas/11/dataset/',  # 这是 raw path
									  vae_path='/workspace/codes/othercodes/vae_checkpoints/unchanged_smiles_inuse/checkpoints/080_NOCHANGE_evenhigherkl.ckpt',
									  save_path='/data/')###


	# train,valid = random_split(dataset,lengths=[int(len(dataset)*0.9),int(len(dataset)-int(len(dataset)*0.9))])
	# data = train[0]
	def split(dataset, split_file):
		split_by_name = torch.load(split_file)
		split = {
			k: [dataset.name2id[n] for n in names if n in dataset.name2id]
			for k, names in split_by_name.items()
		}
		subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
		return dataset, subsets


	dataset, subsets = split(dataset, '/dataset/crossdock/crossdocked_pocket10/split_by_name.pt')
	train, val = subsets['train'], subsets['test']
	print(len(dataset), len(train), len(val))

	follow_batch = ['protein_pos', 'ligand_pos']
	j = 0
	loader = DataLoader(train, batch_size=1, follow_batch=follow_batch)
	loader2 = DataLoader(val, batch_size=1, follow_batch=follow_batch)
	for batch in loader:
		print(batch)
		tgt = batch.target
		j += 1
		break
# for batch2 in loader2:
#     print(batch2)
#     j+=1
#     break
# print(j)