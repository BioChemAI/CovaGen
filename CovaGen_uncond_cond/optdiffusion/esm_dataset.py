import os, sys
import random
import string
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



class SequenceLigandPairDataset(Dataset):

	def __init__(self, raw_path, vae_path, save_path, transform=None,sequence=None,data_path=None):
		super().__init__()#
		self.csv_path = "./Models/cross_docked_seqs_test.csv"
		self.transform = transform
		self.db = None
		self.keys = None
		self.vae_path = "./Models/080_NOCHANGE_evenhigherkl.ckpt"
		if sequence is not None:
			characters = string.ascii_letters + string.digits
			random_string = ''.join(random.choice(characters) for _ in range(5))
			self.processed_path = os.path.join(save_path, random_string + ".lmdb")
			self._process_sequence(sequence=sequence)
		else:
			# self.raw_path = raw_path.rstrip('/')  ###
			# self.index_path = os.path.join(self.raw_path, 'crossdocked_pocket10/', 'index.pkl')
			# self.data_path = 'crossdocked_pocket10_processed_esm_preencoded_klhigher80_smi_test_2.lmdb'
			# self.data_path = 'crossdocked_pocket10_processed_esm_preencoded_klhigher80_smi_full_FULL15_tryall.lmdb'
			self.processed_path = './Models/crossdocked_pocket10_processed_esm_preencoded_klhigher80_smi_test_2.lmdb'
			# print("Data is :", self.data_path)
			# if data_path is not None:
			# 	self.processed_path = os.path.join(save_path,self.data_path)

			# if not os.path.exists(self.processed_path):
			# 	self._process()#

	def _connect_db(self):
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

	def _process_sequence(self,sequence):
		processed_data = []
		success = 0
		smi_ls = []
		esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
		batch_converter = esm_alphabet.get_batch_converter()
		esm_model.eval()

		vae = RNNAttn(load_fn=self.vae_path)
		fail = 0
		df = pd.read_csv(self.csv_path)
		cnt = 0
		result_dict = df.groupby('pocket sequence')['smiles'].apply(list).to_dict()
		for key, value in result_dict.items():
			value_nodup = list(set(value))
			key2 = sequence
			key1 = [("protein1", key2)]
			batch_labels, batch_strs, batch_tokens = batch_converter(key1)
			batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)
			with torch.no_grad():
				results = esm_model(batch_tokens, repr_layers=[6], return_contacts=True)
			token_representations = results["representations"][6]
			print(token_representations.shape)
			sequence_representations = []
			for i, tokens_len in enumerate(batch_lens):
				sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
			for smi in value_nodup:
				# try:
				# 	selfies_rep = sf.encoder(smi)
				# except:
				# 	fail += 1
				# 	print(f"success:{success},fail:{fail}")
				# 	continue
				smiles_emb = torch.zeros(128)
				# if smiles_emb is None:
				# 	fail += 1
				# 	continue
				# smi_ls.append(smiles_emb)

				data = {'seq': key, 'smiles_emb': smiles_emb, 'smiles': "C", 'token_rep': token_representations,
				        'seq_rep': sequence_representations,
				        }
				processed_data.append(data)
				success += 1
				print(f"success:{success},fail:{fail}")
				break
			break

		time.sleep(4)
		db = lmdb.open(
			self.processed_path,
			map_size=10 * (1024 * 1024 * 1024),
			create=True,
			subdir=False,
			readonly=False,
		)
		with db.begin(write=True, buffers=True) as txn:
			for idx, data in enumerate(processed_data):
				txn.put(
					key=str(idx).encode(),
					value=pickle.dumps(data)
				)
		db.close()

	def _process(self):
			fail = 0
			success = 0
			smi_fail = 0
			processed_data = []
			esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
			batch_converter = esm_alphabet.get_batch_converter()
			esm_model.eval()

			vae = RNNAttn(load_fn=self.vae_path)
			df = pd.read_csv(self.csv_path)
			seqls = df['pocket sequence'].to_list()
			smils = df['smiles'].to_list()

			pbar = tqdm(range(len(seqls)))


			db = lmdb.open(
				self.processed_path,
				map_size=41 * (1024 * 1024 * 1024),  # 10GB
				create=True,
				subdir=False,
				readonly=False,  # Writable
			)

			try:
				cnt = 0
				for i in pbar:
					if cnt % 1000 == 0 and cnt > 0:
						print(f"Writing batch to LMDB, processed {cnt} entries so far...")
						with db.begin(write=True) as txn:
							for idx, data in enumerate(processed_data):
								txn.put(
									key=str(cnt + idx).encode(),
									value=pickle.dumps(data)
								)
						processed_data = []

					indd = i
					value_nodup = [smils[indd]]
					key1 = [('protein1', seqls[indd])]

					batch_labels, batch_strs, batch_tokens = batch_converter(key1)
					batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)

					with torch.no_grad():
						results = esm_model(batch_tokens, repr_layers=[6], return_contacts=True)
					token_representations = results["representations"][6]

					for smi in value_nodup:
						try:
							smiles_emb = smiles_to_embed(smi, vae_model=vae)
						except Exception as e:
							fail += 1
							print("Occurred:", e)
							print(f"Success: {success}, Fail: {fail}")
							continue

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
				db.close()


	def _precompute_name2id(self):
		name2id = {}
		for i in tqdm(range(self.__len__()), 'Indexing'):
			try:
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
		key = self.keys[idx]
		data = pickle.loads(self.db.begin().get(key))
		# return data
		sample  =data['token_rep'].squeeze(0)
		padded_sample = torch.nn.functional.pad(torch.tensor(sample), (0, 0, 0, 640 - len(sample)))

		mask = torch.zeros(640, dtype=torch.bool)
		mask[:len(sample)] = 1

		return padded_sample, mask, data['smiles_emb']

if __name__ == '__main__':
	from torch_geometric.transforms import Compose
	from torch_geometric.data import DataLoader
	from torch.utils.data import random_split
	from torch.utils.data import Subset
	import sys

	sys.path.append('..')
	import os
	device = torch.device('cuda:0')
	split_by_name = torch.load('/workspace/datas/crossdock_protein/crossdocked_pocket10/split_by_name.pt')
	dataset = SequenceLigandPairDataset('/workspace/datas/11/dataset/',
									  vae_path='/workspace/codes/othercodes/vae_checkpoints/unchanged_smiles_inuse/checkpoints/080_NOCHANGE_evenhigherkl.ckpt',
									  save_path='/data/')###
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