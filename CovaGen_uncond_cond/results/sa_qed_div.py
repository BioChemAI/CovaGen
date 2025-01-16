import torch
import random
import numpy as np
import pandas as pd
import csv
import math
import pickle
from rdkit.Chem import Descriptors
from rdkit import Chem,DataStructs
from rdkit.Chem import QED
from rdkit.Chem import rdMolDescriptors
import os
import torch.nn.functional as F
import os.path as op
from tqdm.auto import tqdm
import selfies as sf
from itertools import combinations
from rdkit.rdBase import BlockLogs
import argparse
# get_sa_score start
_fscores = None

def readFragmentScores(name='fpscores'):
	import gzip
	global _fscores
	# generate the full path filename:
	if name == "fpscores":
		name = op.join(os.getcwd(), name)
	# name = op.join(op.dirname(__file__), name)
	data = pickle.load(gzip.open('%s.pkl.gz' % name))
	outDict = {}
	for i in data:
		for j in range(1, len(i)):
			outDict[i[j]] = float(i[0])
	_fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
	nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
	nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
	return nBridgehead, nSpiro

def calculateScore(m):
	if _fscores is None:
		readFragmentScores()

	# fragment score
	fp = rdMolDescriptors.GetMorganFingerprint(m,
											   2)  # <- 2 is the *radius* of the circular fingerprint
	fps = fp.GetNonzeroElements()
	score1 = 0.
	nf = 0
	for bitId, v in fps.items():
		nf += v
		sfp = bitId
		score1 += _fscores.get(sfp, -4) * v
	score1 /= nf

	# features score
	nAtoms = m.GetNumAtoms()
	nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
	ri = m.GetRingInfo()
	nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
	nMacrocycles = 0
	for x in ri.AtomRings():
		if len(x) > 8:
			nMacrocycles += 1

	sizePenalty = nAtoms ** 1.005 - nAtoms
	stereoPenalty = math.log10(nChiralCenters + 1)
	spiroPenalty = math.log10(nSpiro + 1)
	bridgePenalty = math.log10(nBridgeheads + 1)
	macrocyclePenalty = 0.
	# ---------------------------------------
	# This differs from the paper, which defines:
	# macrocyclePenalty = math.log10(nMacrocycles+1)
	# This form generates better results when 2 or more macrocycles are present
	if nMacrocycles > 0:
		macrocyclePenalty = math.log10(2)

	score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

	# correction for the fingerprint density
	# not in the original publication, added in version 1.1
	# to make highly symmetrical molecules easier to synthetise
	score3 = 0.
	if nAtoms > len(fps):
		score3 = math.log(float(nAtoms) / len(fps)) * .5

	sascore = score1 + score2 + score3

	# need to transform "raw" value into scale between 1 and 10
	min = -4.0
	max = 2.5
	sascore = 11. - (sascore - min + 1) / (max - min) * 9.
	# smooth the 10-end
	if sascore > 8.:
		sascore = 8. + math.log(sascore + 1. - 9.)
	if sascore > 10.:
		sascore = 10.0
	elif sascore < 1.:
		sascore = 1.0

	return sascore

def smiles_to_fp(smiles: str):
	block = BlockLogs()
	m = Chem.MolFromSmiles(smiles)
	del block
	return Chem.RDKFingerprint(m) if m else None

def smiles_list_to_fps(smiles_list: str, clean=True):
	fps = [smiles_to_fp(x) for x in smiles_list]
	return [x for x in fps if x] if clean else fps

def get_similarity_comb(fps):
	# Takes in a list of fingerprints
	similarity_list = [
		DataStructs.FingerprintSimilarity(fp_x, fp_y) \
			for fp_x, fp_y in combinations(fps, 2)
	]
	return np.mean(similarity_list)

def fraction_unique(gen, k=None, n_jobs=1):
	canon = []
	for i in gen:
		canon.append(Chem.MolToSmiles(i))
	canonic = set(canon) # 转canon
	return len(canonic) / len(gen)

def novelty(gen, train, n_jobs=1):
	canon = []
	for i in gen:
		canon.append(Chem.MolToSmiles(i))
	gen_smiles_set = set(canon) - {None}
	train_set = set(train)
	return len(gen_smiles_set - train_set) / len(gen_smiles_set)

def get_all_score(mols: list, save_name=None, save_path=None):
	# Takes in a list of mols, calcute qed,sa,weight,lop of them. Also calculate the diversity.
	train_ls = []
	readFragmentScores("fpscores")
	with open("/workspace/codes/vaemodel/zinc_128merged_withcrossdock.txt", 'r') as f:
		for line in f:
			train_ls.append(line.rstrip())
		# Get similarity then calculate diversity with it.
	fps=[Chem.RDKFingerprint(m) for m in mols]
	inner_similarity = get_similarity_comb(fps)
	inner_diversity =1-inner_similarity
	unique = fraction_unique(mols,1000)
	nov = novelty(mols,train_ls)
	# Get sa,qed,weight,logp. Return
	tot = []
	tot_qed = []
	tot_wt = []
	tot_logp = []
	# pbar = tqdm(mols)
	for i,m in enumerate(mols):
		s = calculateScore(m)
		s = round((10 - s) / 9, 2)
		try:
			qeds = QED.qed(m)
		except:
			qeds = 0
		wt = Descriptors.MolWt(m)
		lgp = Chem.Crippen.MolLogP(m)
		smiles = Chem.MolToSmiles(m)
		# print(smiles + "\t" + "\t%3f" % s + "\t" + "\t%3f" % qeds+ "\t" + "\t%3f")
		tot.append(s)
		tot_qed.append(qeds)
		tot_wt.append(wt)
		tot_logp.append(lgp)
	tot2 = np.array(tot)
	tot_qed2 = np.array(tot_qed)
	tot_wt2 = np.array(tot_wt)
	tot_logp2 = np.array(tot_logp)
	# print("SA:mean:{}, min:{}",tot/len(mols))
	# print("mean qed",totq/len(mols))
	print(f"SA : mean:{np.mean(tot2)}, median:{np.median(tot2)}, min:{np.min(tot2)}, max:{np.max(tot2)}")
	print(f"QED : mean:{np.mean(tot_qed2)}, median:{np.median(tot_qed2)}, min:{np.min(tot_qed2)}, max:{np.max(tot_qed2)}")
	print(f"weight : mean:{np.mean(tot_wt2)}, median:{np.median(tot_wt2)}, min:{np.min(tot_wt2)}, max:{np.max(tot_wt2)}")
	print(f"LogP : mean:{np.mean(tot_logp2)}, median:{np.median(tot_logp2)}, min:{np.min(tot_logp2)}, max:{np.max(tot_logp2)}")
	print("Diversity :", inner_diversity)
	print("unique@1000 :", unique)
	print("Novelty :",nov )
	# return tot,tot_qed,tot_wt,tot_logp,inner_diversity

	filename =save_path+save_name+'.csv'

	# 创建 CSV 文件并写入数据
	with open(filename, "w", newline="") as csvfile:
		writer = csv.writer(csvfile)

		# 写入每一列的标题
		writer.writerow(["sa", "qed", "weight", "logp"])

		# 写入数据
		for i in range(len(tot)):
			row = [tot[i], tot_qed[i], tot_wt[i], tot_logp[i]]
			writer.writerow(row)

if __name__ == '__main__':##
	parser = argparse.ArgumentParser()
	parser.add_argument("--molecules", type=str)
	parser.add_argument("--save_path", type=str)
	parser.add_argument("--save_name", type=str)
	args = parser.parse_args()
	with open(args.molecules,'rb')as f: ##
		k = pickle.load(f)
	smis = []
	for i in k:
		smi = sf.decoder(i)
		smis.append(smi)
	mols = []
	for i in smis:
		mol = Chem.MolFromSmiles(i)
		if mol!=None:
			mols.append(mol)
	get_all_score(mols,save_name=args.save_name,save_path=args.save_path)
	print('done')
