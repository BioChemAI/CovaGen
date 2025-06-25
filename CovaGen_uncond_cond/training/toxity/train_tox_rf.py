import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools
def smiles_to_one_hot(smiles_string):
	max_length = 128
	smiles_dict = {
		0: 'C', 1: 'N', 2: '(', 3: '=', 4: 'O', 5: ')', 6: 'c', 7: '1', 8: '2', 9: '#',
		10: 'n', 11: 'Cl', 12: '-', 13: '3', 14: 'o', 15: 'Br', 16: '[', 17: '@', 18: 'H',
		19: ']', 20: 's', 21: '4', 22: 'B', 23: 'F', 24: 'S', 25: '5', 26: 'I', 27: '6',
		28: '/', 29: 'i', 30: '+', 31: '\\', 32: 'P', 33: '7', 34: 'Z', 35: 'r', 36: 'M',
		37: 'g', 38: 'L', 39: 'f', 40: 'T', 41: 'e', 42: 'K', 43: 'V', 44: 'A', 45: 'l',
		46: 'b', 47: '8', 48: '9', 49: 'a', 50: 't', 51: 'Y', 52: 'G', 53: 'R', 54: 'u',
		55: 'p', 56: 'h', 57: 'U', 58: 'd', 59: 'W', 60: '%', 61: '0', 62: 'X', 63: '_',
	}
	num_chars = len(smiles_dict)
	one_hot_vector = np.zeros((max_length, num_chars))
	for i, char in enumerate(smiles_string):
		if i >= max_length:
			break
		if char in smiles_dict.values():
			char_index = [k for k, v in smiles_dict.items() if v == char][0]
			one_hot_vector[i, char_index] = 1
	return one_hot_vector

with open('/workspace/codes/toxicity/hepato_emb.pkl','rb')as f:
	data_list = pickle.load(f)
with open('/workspace/codes/toxicity/hepato_tag.pkl','rb')as f:
	label_list = pickle.load(f)

if __name__ == '__main__':

	# 划分训练集和测试集
	# csv1 = pd.read_csv('/workspace/codes/toxicity/Hepato.csv')
	# # with open()
	# csv1 = csv1.dropna(subset=['Canonical SMILES'])
	# data_list=csv1['Canonical SMILES'].tolist()
	# cleaned_list = [x for x in data_list if isinstance(x,str)]
	# cleaned_list = [smiles.split('.')[0] for smiles in cleaned_list]
	# label_list=csv1['Toxicity Value'].tolist()
	# # hot_list = []
	# # for i in data_list:
	# #     hot = smiles_to_one_hot(i).flatten()
	# #     hot_list.append(hot)
	# molecules = [Chem.MolFromSmiles(smiles) for smiles in cleaned_list]
	# #
	# # 计算MACCS指纹
	# maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules]
	# fp_array = np.array([list(fp) for fp in maccs_fps])
	# fps = [pyAvalonTools.GetAvalonFP(mol) for mol in molecules]
	# fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useFeatures=True) for mol in molecules]
	# fp_array = np.array(fps)
	da = []
	data_list = np.stack(data_list)
	X_train, X_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.1, random_state=42)

	# 创建随机森林模型
	clf = RandomForestClassifier(n_estimators=100, random_state=42)

	# 在训练集上训练模型
	clf.fit(X_train, y_train)

	# 在测试集上进行预测
	y_pred = clf.predict(X_test)

	# 计算准确率
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {accuracy:.2f}")
	with open('/workspace/codes/toxicity/rf_emb.pkl', 'wb') as model_file:
		pickle.dump(clf, model_file)