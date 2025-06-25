import pickle

from rdkit import Chem
from rdkit.Chem import Draw,AllChem
from rdkit.Chem import QED
from rdkit.Chem import MACCSkeys

import torch
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import selfies as sf
def smiles_to_maccs(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	fp = MACCSkeys.GenMACCSKeys(mol)
	return np.array([int(fp[i]) for i in range(1, fp.GetNumBits())], dtype=int)

def check_hepatotoxicity(smi_list, model_path='/workspace/codes/lddd_ddpo/guided_diffusion/rf_maccs_model2.pkl'):
	# 加载保存的 pipeline 模型（含随机森林）
	with open(model_path, 'rb') as f:
		model = pickle.load(f)

	# 转换 SMILES 到 MACCS 指纹
	fps = []
	valid_indices = []
	for i, smi in enumerate(smi_list):
		fp = smiles_to_maccs(smi)
		if fp is not None:
			fps.append(fp)
			valid_indices.append(i)

	if not fps:
		raise ValueError("输入的 SMILES 中无有效结构")

	X = np.stack(fps)

	# 预测毒性概率（类别 1 表示有肝毒性）
	probs = model.predict_proba(X)[:, 1]  # 有毒的概率
	reward = (1.0 - probs) * 10.0         # 无毒性 → 高 reward（0 ~ 10）
	avg_toxicity = np.mean(probs)
	print(f"平均肝毒性概率: {avg_toxicity:.4f}")
	# 将结果放回原始顺序，非法 SMILES 对应 reward 为 None
	full_reward = [None] * len(smi_list)
	for idx, r in zip(valid_indices, reward):
		full_reward[idx] = r

	return torch.tensor(full_reward)

def check_cardiotoxicity(smi_list, model_path='/workspace/codes/lddd_ddpo/guided_diffusion/rf_maccs_model_cardio.pkl'):
	# 加载保存的 pipeline 模型（含随机森林）
	with open(model_path, 'rb') as f:
		model = pickle.load(f)

	# 转换 SMILES 到 MACCS 指纹
	fps = []
	valid_indices = []
	for i, smi in enumerate(smi_list):
		fp = smiles_to_maccs(smi)
		if fp is not None:
			fps.append(fp)
			valid_indices.append(i)

	if not fps:
		raise ValueError("输入的 SMILES 中无有效结构")

	X = np.stack(fps)

	# 预测毒性概率（类别 1 表示有肝毒性）
	probs = model.predict_proba(X)[:, 1]  # 有毒的概率
	reward = (1.0 - probs) * 10.0         # 无毒性 → 高 reward（0 ~ 10）
	avg_toxicity = np.mean(probs)
	print(f"平均肝毒性概率: {avg_toxicity:.4f}")
	# 将结果放回原始顺序，非法 SMILES 对应 reward 为 None
	full_reward = [None] * len(smi_list)
	for idx, r in zip(valid_indices, reward):
		full_reward[idx] = r

	return torch.tensor(full_reward)


#
# def check_hepatotoxicity(smi_list):
#     # 加载训练好的 RF 模型
#     with open('/workspace/codes/lddd_ddpo/guided_diffusion/rf_maccs_model2.pkl', 'rb') as f:
#         rf_model = pickle.load(f)
#
#     # SMILES → 分子
#     molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list]
#     valid_molecules = [mol for mol in molecules if mol is not None]#
#
#     # MACCS 指纹提取
#     fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in valid_molecules]
#     fp_array = np.array([list(fp) for fp in fps])
#
#     # 预测肝毒性概率
#     probs = rf_model.predict_proba(fp_array)[:, 1]  # 类别1表示“有毒”的概率
#
#     # 转换为 reward（概率越低越好）
#     reward = (1.0 - probs) * 10.0  # 映射到 0~10 区间
#
#     return reward


def check_toxicity_xgb(smi_list):
	with open('/workspace/codes/emiddpm/toxity/xgboost_model_maccs.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list]
	maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
	fp_array = np.array([list(fp) for fp in maccs_fps])
	dtest = xgb.DMatrix(fp_array)
	y_pred_loaded = xgb_model.predict(dtest)##
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())
	rescaled_values = np.exp(y_pred_loaded)

	# 进一步缩放到你想要的范围（例如，0到10）
	# min_go = np.min(rescaled_values)
	# max_go = np.max(rescaled_values)
	# min_go = 1
	# max_go
	min_output = 0
	max_output = 10
	rescaled_values = (rescaled_values - np.min(rescaled_values)) / (
				np.max(rescaled_values) - np.min(rescaled_values)) * (max_output - min_output) + min_output
	reversed_val = [10-i for i in rescaled_values]



	return torch.tensor(reversed_val)

def check_toxicity_xgb_old_2(smi_list):
	with open('/workspace/codes/emiddpm/toxity/xgboost_model_maccs.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list if Chem.MolFromSmiles(smiles) is not None]
	maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
	fp_array = np.array([list(fp) for fp in maccs_fps])
	dtest = xgb.DMatrix(fp_array)
	y_pred_loaded = xgb_model.predict(dtest)
	# y_pred_loaded = xgb_model.predict(fp_array)  ##
	print("mean tox", y_pred_loaded.mean())
	print("min tox", y_pred_loaded.min())

	min_value = 2.25
	max_value = 2.85
	toxicity_values = np.array(y_pred_loaded)
	# 定义目标得分范围
	# modified_array = np.where(toxicity_values < 2.2, 2.0, np.where(toxicity_values > 2.8, 3.2, toxicity_values))
	min_score = 5
	max_score = 0

	# 利用线性映射计算得分
	scores = (toxicity_values - min_value) / (max_value - min_value) * (max_score - min_score) + min_score

	wt = []
	# 分子量部分

	# 将得分限制在0到5的范围内

	for i in range(len(scores)):  # 似乎是有点问题，怎么主跑这个的时候就会在descriptors这里报错？
		mol_weight = Chem.Descriptors.MolWt(molecules[i])
		wt.append((mol_weight - 200) / 100)
		if mol_weight >= 230 and toxicity_values[i] <= 2.45:
			scores[i] += 4
		if mol_weight <= 175:
			scores[i] -= 6
		wt.append(mol_weight)
	# wt=np.array(wt)
	# scores = scores+wt
	scores = np.clip(scores, -4, 6)
	print("mean weight:", np.array(wt).mean())

	return torch.tensor(scores)



def check_toxicity_xgb_2(smi_list):
	# with open('/workspace/codes/emiddpm/toxity/xgboost_model_maccs.pkl', "rb") as model_file:
	# 	xgb_model = pickle.load(model_file)
	with open('/workspace/codes/emiddpm/toxity/tpot_toxicity_best_model.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list if Chem.MolFromSmiles(smiles) is not None]
	maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
	fp_array = np.array([list(fp) for fp in maccs_fps])
	# dtest = xgb.DMatrix(fp_array)
	y_pred_loaded = xgb_model.predict(fp_array)##
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())

	min_value = 2.25
	max_value = 2.85
	toxicity_values = np.array(y_pred_loaded)
	# 定义目标得分范围
	# modified_array = np.where(toxicity_values < 2.2, 2.0, np.where(toxicity_values > 2.8, 3.2, toxicity_values))
	min_score = 5
	max_score = 0

	# 利用线性映射计算得分
	scores = (toxicity_values - min_value) / (max_value - min_value) * (max_score - min_score) + min_score

	wt = []
	# 分子量部分

	# 将得分限制在0到5的范围内


	for i in range(len(scores)): # 似乎是有点问题，怎么主跑这个的时候就会在descriptors这里报错？
		mol_weight = Chem.Descriptors.MolWt(molecules[i])
		wt.append((mol_weight - 200) / 100)
		if mol_weight>=220 and toxicity_values[i]<=2.5:
			scores[i]+=5
		if mol_weight<=175:
			scores[i]-=6
		wt.append(mol_weight)
	# wt=np.array(wt)
	# scores = scores+wt
	scores = np.clip(scores, -6, 6)
	print("mean weight:", np.array(wt).mean())

	return torch.tensor(scores)

def check_toxicity_xgb_3(smi_list):
	"""
	Removed MW penelisation, 给invalid者一个负分。
	Args:
		smi_list:

	Returns:

	"""
	# with open('/workspace/codes/emiddpm/toxity/xgboost_model_maccs.pkl', "rb") as model_file:
	# 	xgb_model = pickle.load(model_file)
	cnt_ls = []
	crct_ls = []
	for cnt, smis in enumerate(smi_list):
		mol = Chem.MolFromSmiles(smis)
		if mol is None:
			cnt_ls.append(cnt)
		else:
			crct_ls.append(smis)
	with open('/workspace/codes/emiddpm/toxity/tpot_toxicity_best_model.pkl', "rb") as model_file:
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
	# dtest = xgb.DMatrix(fp_array)

	y_pred_loaded = xgb_model.predict(fp_array)##
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())
	# scalar版本
	# scaler = MinMaxScaler(feature_range=(-3, 3))
	# toxicity_values = toxicity_values.reshape(-1, 1)
	# scores = scaler.fit_transform(toxicity_values)
	# scores = -scores.squeeze()

	min_value = 2.25
	max_value = 2.85
	toxicity_values = np.array(y_pred_loaded)
	# 定义目标得分范围
	# modified_array = np.where(toxicity_values < 2.2, 2.0, np.where(toxicity_values > 2.8, 3.2, toxicity_values))
	min_score = 5
	max_score = 0

	# 利用线性映射计算得分
	scores = (toxicity_values - min_value) / (max_value - min_value) * (max_score - min_score) + min_score

	wt = []
	# 分子量部分
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
	# 将得分限制在0到5的范围内

	for i in cnt_ls:
		scores[i] = -10
	print("mean weight:", np.array(wt).mean())

	return torch.tensor(scores)

def check_toxicity_xgb_SELFIES(smi_list):
	"""
	Removed MW penelisation, 给invalid者一个负分。
	Args:
		smi_list:

	Returns:

	"""
	# with open('/workspace/codes/emiddpm/toxity/xgboost_model_maccs.pkl', "rb") as model_file:
	# 	xgb_model = pickle.load(model_file)
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
	with open('/workspace/codes/emiddpm/toxity/tpot_toxicity_best_model.pkl', "rb") as model_file:
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
	# dtest = xgb.DMatrix(fp_array)

	y_pred_loaded = xgb_model.predict(fp_array)##
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())

	# min_value = 2.25
	# max_value = 2.85
	toxicity_values = np.array(y_pred_loaded)

	scaler = MinMaxScaler(feature_range=(-3, 20))
	# 利用线性映射计算得分
	toxicity_values = toxicity_values.reshape(-1, 1)
	scores = scaler.fit_transform(toxicity_values)
	scores = -scores.squeeze()
	# 定义目标得分范围
	# modified_array = np.where(toxicity_values < 2.2, 2.0, np.where(toxicity_values > 2.8, 3.2, toxicity_values))
	min_score = 5
	max_score = 0

	# 利用线性映射计算得分
	# scores = (toxicity_values - min_value) / (max_value - min_value) * (max_score - min_score) + min_score

	wt = []
	# 分子量部分

	# 将得分限制在0到5的范围内


	# wt=np.array(wt)
	# scores = scores+wt
	# scores = np.clip(scores, -3, 4)
	for i in cnt_ls:
		scores[i] = -10
	print("mean weight:", np.array(wt).mean())

	return torch.tensor(scores)

def check_toxicity_xgb_new(smi_list):
	# with open('/workspace/codes/emiddpm/toxity/xgboost_model_maccs.pkl', "rb") as model_file:
	# 	xgb_model = pickle.load(model_file)
	with open('/workspace/codes/emiddpm/toxity/tpot_toxicity_best_model.pkl', "rb") as model_file:
		xgb_model = pickle.load(model_file)
	molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list]
	maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
	fp_array = np.array([list(fp) for fp in maccs_fps])
	# dtest = xgb.DMatrix(fp_array)
	y_pred_loaded = xgb_model.predict(fp_array)##
	print("mean tox", y_pred_loaded.mean())
	print("min tox",y_pred_loaded.min())
	min_value = 2.25
	max_value = 2.85
	toxicity_values = np.array(y_pred_loaded)
	# toxicity_values = np.clip(toxicity_values, 1.9, 2.7)
	# 定义目标得分范围
	min_score = 5
	max_score = 0
	scaler = MinMaxScaler(feature_range=(-3, 3))
	# 利用线性映射计算得分
	toxicity_values = toxicity_values.reshape(-1, 1)
	scores = scaler.fit_transform(toxicity_values)
	scores = -scores
	wt = []
	# 分子量部分

	# 将得分限制在0到5的范围内

	scores = np.squeeze(scores)
	for i in range(len(scores)):
		mol_weight = Chem.Descriptors.MolWt(molecules[i])
		wt.append((mol_weight - 200) / 100)
		if mol_weight>=220 and toxicity_values[i]<=2.25:
			scores[i]+=3
		if mol_weight<=160:
			scores[i]-=2
		wt.append(mol_weight)
	# wt=np.array(wt)
	# scores = scores+wt
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

# 输入你的分子SMILES表示法（或其他格式的分子描述）
def check_ben(smiles): # 这是苯环的SMILES表示法示例
	'''
	Take in a LIST of smiles
	Args:
		smiles:

	Returns:

	'''
	# 将SMILES字符串转换为RDKit分子对象
	score_list = []
	for i in smiles:
		mol = Chem.MolFromSmiles(i)

		# 检查分子是否包含苯环
		if mol:
			# 通过RDKit的SubstructMatch函数来检查苯环的存在
			benzene_pattern = Chem.MolFromSmiles("c1ccccc1")  # 苯环的SMILES表示法
			if mol.HasSubstructMatch(benzene_pattern):
				score_list.append(0.00001)
			else:
				score_list.append(10.00001)
	filtered_list = [x for x in score_list if x > 1.05]
	count_of_elements_greater_than_1_05 = len(filtered_list)
	print('benes:', count_of_elements_greater_than_1_05)
	return torch.tensor(score_list)

if __name__ == '__main__':
	with open("/workspace/codes/grid_search/dec_ddpo_notanh_all_8_esm.pkl",'rb') as f:
		k = pickle.load(f)
	new_cleaned = []
	# for i in k:
	# 	# i = random.sample(list(set(i)),30)
	# 	new_cleaned.extend(i)
	#
	# cleaned_list = new_cleaned
	# cleaned_list = list(set(cleaned_list))
	mscore = check_toxicity_xgb_2(k)
	print("done")
