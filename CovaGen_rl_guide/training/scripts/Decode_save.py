"""
Decode and save to 10k valid and all
"""
from rdkit import Chem
import pickle
import torch
import numpy as np
import os,sys
sys.path.append(os.path.dirname(sys.path[0]))
from transvae.rnn_models import RNNAttn
import argparse
from datetime import datetime, date

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--sampled_vec", type=str)
	parser.add_argument("--num_samples", type=int)
	parser.add_argument('--save_path_10k', type=str)
	parser.add_argument('--save_path_full', type=str)
	parser.add_argument('--vae_path', type=str)
	args = parser.parse_args()

	vec_dic = {}

	for i in range(0,100):
		with open(args.sampled_vec+'/'+f'samp{i}th.pkl',"rb") as f:
			vec_dic[f"{i}"] = pickle.load(f)

	toredu = []
	smi_ls = []
	count = 0
	i = 0
	tmpcnt=0
	crct_ls = []
	k=0
	vae = RNNAttn(load_fn=args.vae_path)
	time_0 = datetime.now()
	for key,value in vec_dic.items():
		for i in range(0,args.num_samples,500):
			data = value[i:i + 500]
			data = torch.tensor(np.stack(data)).to('cuda:0')
			smi_recon = vae.sample2(data)
			smi_ls.extend(smi_recon)
			for smis in smi_recon:
				mol = Chem.MolFromSmiles(smis)
				if mol is not None:
					count += 1
					crct_ls.append(smis)
					tmpcnt += 1
					if tmpcnt == 100:
						break
			if tmpcnt == 100:
				tmpcnt = 0
				break
	time_1 = datetime.now()
	seconds = (time_1 - time_0).seconds
	print('time:',seconds)
	with open(args.save_path_10k,"wb") as f	:
		pickle.dump(crct_ls, f)

	with open(args.save_path_full,"wb")as f:
		pickle.dump(smi_ls,f)
	print("done")