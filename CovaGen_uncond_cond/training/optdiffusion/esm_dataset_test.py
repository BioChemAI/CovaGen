import os, sys
import random

import pandas as pd
import functools
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
		super().__init__()
		self.csv_path = "/workspace/datas/cross_docked_seqs_train.csv"

		self.raw_path = raw_path.rstrip('/')  ###
		self.index_path = os.path.join(self.raw_path, 'crossdocked_pocket10/', 'index.pkl')
		self.processed_path = os.path.join(save_path,
										   'crossdocked_pocket10_processed_esm_preencoded_klhigher80_smi_train_full.lmdb')  # 这个就是lmdb那里那个db连接到的数据的位置！
		self.transform = transform
		self.db = None
		self.keys = None
		self.vae_path = vae_path

		self.fail=0
		self.success=0
		self.smi_fail=0



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
			map_size= 30 * (1024 * 1024 * 1024),  # 10GB
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

	def heapply(self,row,batch_converter=None,esm_alphabet=None,esm_model=None,processed_data=None,vae=None):
		value_nodup = [row['smiles']]
		key1 = row['pocket sequence']  #
		key1 = [('protein1', key1)]  #
		batch_labels, batch_strs, batch_tokens = batch_converter(key1)
		batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)
		# batch_tokens = batch_tokens.cuda()
		with torch.no_grad():
			results = esm_model(batch_tokens, repr_layers=[6], return_contacts=True)
		token_representations = results["representations"][6]
		# print(token_representations.shape)
		# Generate per-sequence representations via averaging
		# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
		sequence_representations = []
		for i, tokens_len in enumerate(batch_lens):
			sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
		for smi in value_nodup:
			# try:
			# 	selfies_rep = sf.encoder(smi)
			# except Exception as e:
			# 	fail+=1
			# 	print("occured:",e)
			# 	print(f"success:{success},fail:{fail}")
			# 	continue
			try:
				smiles_emb = smiles_to_embed(smi, vae_model=vae)
			except Exception as e:
				self.fail += 1
				print("occured:", e)
				print(f"success:{self.success},fail:{self.fail}")
				continue  # smiles emb用的就是vae输出的mu。
			if smiles_emb is None:
				self.fail += 1
				self.smi_fail += 1
				print("encode fail:",self.smi_fail)
				continue  # 那估计是字典不够大了。。
			# smi_ls.append(smiles_emb)

			data = {'seq': row['pocket sequence'], 'smiles_emb': smiles_emb, 'smiles': smi,
					'token_rep': token_representations, 'seq_rep': sequence_representations,
					}  # 最后获得的data的是一个dict，重点的是其中的pocket和smiles_emb。我的理解是一个pocket可以有不同种类的ligand和其结合，这样可以对pocket取一个dict来形成这个
			# 一对多的关系。
			# print(data)
			processed_data.append(data)
			self.success += 1
			print(f"success:{self.success},fail:{self.fail}")

	def _process(self):
		processed_data = []
		fail = 0
		success = 0
		smi_ls=[]

		esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
		batch_converter = esm_alphabet.get_batch_converter()
		esm_model.eval()
		# esm_model = esm_model.cuda()
		vae = RNNAttn(load_fn=self.vae_path)
		fail = 0
		smi_fail = 0
		df = pd.read_csv(self.csv_path)
		# result_dict = df.groupby('pocket sequence')['smiles'].apply(list).to_dict()
		cnt = -1
		original_list = list(range(100000))

		# 从列表中随机抽取50000个互不重复的数字
		random_sample_df = df.sample(n=50000)
		apply_fun = functools.partial(
			self.heapply,  # ！！去深入地看一下代码中对这几种loss的综合处理和输出？？？！！
			batch_converter=batch_converter,
			esm_model=esm_model,
			esm_alphabet=esm_alphabet,
			processed_data=processed_data,
			vae=vae
		)
		random_sample_df.apply(apply_fun,axis=1)


				# if cnt >= 20000:
				# 	break
			# if cnt >= 20000:
			# 	break
		# index = index[:20000] #for dev
		### convert to smiles; remove duplicate

		# save the data

		time.sleep(10)
		db = lmdb.open(
			self.processed_path,
			map_size=30 * (1024 * 1024 * 1024),  # 10GB
			create=True,
			subdir=False,
			readonly=False,  # Writable
		)
		with db.begin(write=True, buffers=True) as txn:
			for idx, data in enumerate(processed_data):
				txn.put(
					key=str(idx).encode(),
					value=pickle.dumps(data)
				)
		db.close()  # 最后采用了lmdb的方法保存的。应该是在这个路径：crossdocked_pocket10_processed.lmdb

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
	split_by_name = torch.load('/workspace/datas/11/split_by_name.pt')
	# dataset = PocketLigandPairDataset('/workspace/stu/ltx/nt/dataset/dataset/', # 这是 raw path
	#                                  vae_path='/workspace/stu/ltx/nt/045_trans1x-128_zinc.ckpt',
	#                                  save_path='/workspace/stu/ltx/nt/dataset/dataset/processed/')
	dataset = SequenceLigandPairDataset('/workspace/datas/11/dataset/',  # 这是 raw path
									  vae_path='/workspace/codes/othercodes/vae_checkpoints/unchanged_smiles_inuse/checkpoints/080_NOCHANGE_evenhigherkl.ckpt',
									  save_path='/data/')##


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