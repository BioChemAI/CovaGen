import pickle

from rdkit import Chem
from rdkit.Chem import Draw,AllChem
from rdkit.Chem import QED

import torch
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import selfies as sf

def check_toxicity_xgb(smi_list):
	with open('./Models/xgboost_model_maccs.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list]
	maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
	fp_array = np.array([list(fp) for fp in maccs_fps])
	dtest = xgb.DMatrix(fp_array)
	y_pred_loaded = xgb_model.predict(dtest)##
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())
	rescaled_values = np.exp(y_pred_loaded)

	min_output = 0
	max_output = 10
	rescaled_values = (rescaled_values - np.min(rescaled_values)) / (
				np.max(rescaled_values) - np.min(rescaled_values)) * (max_output - min_output) + min_output
	reversed_val = [10-i for i in rescaled_values]



	return torch.tensor(reversed_val)

def check_toxicity_xgb_old_2(smi_list):
	with open('./Models/xgboost_model_maccs.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list if Chem.MolFromSmiles(smiles) is not None]
	maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
	fp_array = np.array([list(fp) for fp in maccs_fps])
	dtest = xgb.DMatrix(fp_array)
	y_pred_loaded = xgb_model.predict(dtest)
	print("mean tox", y_pred_loaded.mean())
	print("min tox", y_pred_loaded.min())

	min_value = 2.25
	max_value = 2.85
	toxicity_values = np.array(y_pred_loaded)

	min_score = 5
	max_score = 0
	scores = (toxicity_values - min_value) / (max_value - min_value) * (max_score - min_score) + min_score

	wt = []

	for i in range(len(scores)):
		mol_weight = Chem.Descriptors.MolWt(molecules[i])
		wt.append((mol_weight - 200) / 100)
		if mol_weight >= 230 and toxicity_values[i] <= 2.45:
			scores[i] += 4
		if mol_weight <= 175:
			scores[i] -= 6
		wt.append(mol_weight)
	scores = np.clip(scores, -4, 6)
	print("mean weight:", np.array(wt).mean())

	return torch.tensor(scores)



def check_toxicity_xgb_2(smi_list):

	with open('./Models/tpot_toxicity_best_model.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list if Chem.MolFromSmiles(smiles) is not None]
	maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
	fp_array = np.array([list(fp) for fp in maccs_fps])

	y_pred_loaded = xgb_model.predict(fp_array)##
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())

	min_value = 2.25
	max_value = 2.85
	toxicity_values = np.array(y_pred_loaded)

	min_score = 5
	max_score = 0

	scores = (toxicity_values - min_value) / (max_value - min_value) * (max_score - min_score) + min_score

	wt = []


	for i in range(len(scores)):
		mol_weight = Chem.Descriptors.MolWt(molecules[i])
		wt.append((mol_weight - 200) / 100)
		if mol_weight>=220 and toxicity_values[i]<=2.5:
			scores[i]+=5
		if mol_weight<=175:
			scores[i]-=6
		wt.append(mol_weight)
	scores = np.clip(scores, -6, 6)
	print("mean weight:", np.array(wt).mean())

	return torch.tensor(scores)

def check_toxicity_xgb_3(smi_list):

	cnt_ls = []
	crct_ls = []
	for cnt, smis in enumerate(smi_list):
		mol = Chem.MolFromSmiles(smis)
		if mol is None:
			cnt_ls.append(cnt)
		else:
			crct_ls.append(smis)
	with open('./Models/tpot_toxicity_best_model.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = []

	for smiles in smi_list:
		molecules.append(Chem.MolFromSmiles(smiles))
	maccs_fps = []

	for mol in molecules:
		if mol is not None:
			maccs_fps.append(AllChem.GetMACCSKeysFingerprint(mol))
		else:
			maccs_fps.append(0)

	sup_ls = []
	for i in range(167):
		sup_ls.append(0)
	fpls = []
	for fp in maccs_fps:
		if fp !=0:
			fpls.append(list(fp))
		else:
			fpls.append(sup_ls)

	fp_array = np.array(fpls)


	y_pred_loaded = xgb_model.predict(fp_array)##
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())


	min_value = 2.25
	max_value = 2.85
	toxicity_values = np.array(y_pred_loaded)

	min_score = 5
	max_score = 0

	scores = (toxicity_values - min_value) / (max_value - min_value) * (max_score - min_score) + min_score

	wt = []

	for i in range(len(scores)):
		if molecules[i]!=None:
			mol_weight = Chem.Descriptors.MolWt(molecules[i])
			wt.append((mol_weight - 200) / 100)
			if mol_weight>=220 and toxicity_values[i]<=2.45:
				scores[i]+=3
			if mol_weight<=175:
				scores[i]-=10
			wt.append(mol_weight)
		else:
			continue
	# wt=np.array(wt)
	# scores = scores+wt
	scores = np.clip(scores, -6, 6)

	for i in cnt_ls:
		scores[i] = -10
	print("mean weight:", np.array(wt).mean())

	return torch.tensor(scores)

def check_toxicity_xgb_SELFIES(smi_list):

	cnt_ls = []
	crct_ls = []
	converted = []
	for i in smi_list:
		smi1 = sf.decoder(i)
		converted.append(smi1)
	smi_list = converted
	for cnt, smis in enumerate(smi_list):
		mol = Chem.MolFromSmiles(smis)
		if mol is None:
			cnt_ls.append(cnt)
		else:
			crct_ls.append(smis)
	with open('./Models/tpot_toxicity_best_model.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = []

	for smiles in smi_list:
		molecules.append(Chem.MolFromSmiles(smiles))
	maccs_fps = []

	for mol in molecules:
		if mol is not None:
			maccs_fps.append(AllChem.GetMACCSKeysFingerprint(mol))
		else:
			maccs_fps.append(0)

	sup_ls = []
	for i in range(167):
		sup_ls.append(0)
	fpls = []
	for fp in maccs_fps:
		if fp !=0:
			fpls.append(list(fp))
		else:
			fpls.append(sup_ls)

	fp_array = np.array(fpls)

	y_pred_loaded = xgb_model.predict(fp_array)##
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())

	toxicity_values = np.array(y_pred_loaded)

	scaler = MinMaxScaler(feature_range=(-3, 20))
	toxicity_values = toxicity_values.reshape(-1, 1)
	scores = scaler.fit_transform(toxicity_values)
	scores = -scores.squeeze()
	min_score = 5
	max_score = 0

	wt = []

	for i in cnt_ls:
		scores[i] = -10
	print("mean weight:", np.array(wt).mean())

	return torch.tensor(scores)

def check_toxicity_xgb_new(smi_list):
	with open('./Models/tpot_toxicity_best_model.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list]
	maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
	fp_array = np.array([list(fp) for fp in maccs_fps])
	y_pred_loaded = xgb_model.predict(fp_array)
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())
	min_value = 2.25
	max_value = 2.85
	toxicity_values = np.array(y_pred_loaded)
	min_score = 5
	max_score = 0
	scaler = MinMaxScaler(feature_range=(-3, 3))
	toxicity_values = toxicity_values.reshape(-1, 1)
	scores = scaler.fit_transform(toxicity_values)
	scores = -scores
	wt = []

	scores = np.squeeze(scores)
	for i in range(len(scores)):
		mol_weight = Chem.Descriptors.MolWt(molecules[i])
		wt.append((mol_weight - 200) / 100)
		if mol_weight>=220 and toxicity_values[i]<=2.25:
			scores[i]+=3
		if mol_weight<=160:
			scores[i]-=2
		wt.append(mol_weight)
	scores = np.clip(scores, -4, 8)
	print("mean weight:", np.array(wt).mean())

	return torch.tensor(scores)

def check_qed(smiles):
	score_list = []
	for i in smiles:
		mol = Chem.MolFromSmiles(i)
		qed_value = QED.qed(mol)
		score_list.append(qed_value)
	ls = torch.tensor(score_list)
	print("qed:",ls.mean())
	return ls

def check_ben(smiles):

	score_list = []
	for i in smiles:
		mol = Chem.MolFromSmiles(i)
		if mol:
			benzene_pattern = Chem.MolFromSmiles("c1ccccc1")
			if mol.HasSubstructMatch(benzene_pattern):
				score_list.append(0.00001)
			else:
				score_list.append(10.00001)
	filtered_list = [x for x in score_list if x > 1.05]
	count_of_elements_greater_than_1_05 = len(filtered_list)
	print('benes:', count_of_elements_greater_than_1_05)
	return torch.tensor(score_list)

if __name__ == '__main__':
	# with open("/workspace/codes/grid_search/dec_ddpo_notanh_all_8_esm.pkl",'rb') as f:
	# 	k = pickle.load(f)
	new_cleaned = []
	mscore = check_toxicity_xgb_2(k)
	print("done")
