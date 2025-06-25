import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tpot import TPOTRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
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
print("cnm")
csv1 = pd.read_csv('/workspace/codes/toxicity/Acute_Toxicity_mouse_intraperitoneal_LD50.csv')
# with open()
csv1 = csv1.dropna(subset=['Canonical SMILES'])
smi_list=csv1['Canonical SMILES'].tolist()
tag_list=csv1['mouse_intraperitoneal_LD50'].tolist()
cleaned_list = [x for x in smi_list if isinstance(x,str)]
cleaned_list = [smiles.split('.')[0] for smiles in cleaned_list]
molecules = [Chem.MolFromSmiles(smiles) for smiles in cleaned_list]
	#
	# 计算MACCS指纹
maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules]
fp_array = np.array([list(fp) for fp in maccs_fps])

# hot_list = []
# for i in cleaned_list:
# 	hot_list.append(smiles_to_one_hot(i).flatten())
# hots = np.stack(hot_list)
# with open('/workspace/codes/toxicity/acute_emb_jingmai.pkl', 'rb') as f:
# 	data = pickle.load(f)
# with open('/workspace/codes/toxicity/acute_value_jingmai.pkl','rb')as f:
# 	value = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(fp_array, tag_list, test_size=0.2, random_state=42)
tpot = TPOTRegressor(generations=10, population_size=40, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.fitted_pipeline_)
predictions = tpot.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score0

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
tpot.export('best_regressor_model_4.py')
print(f"均方误差（MSE）: {mse}")
print(f"决定系数（R²）: {r2}")