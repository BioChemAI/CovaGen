import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.metrics import accuracy_score

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

if __name__ == '__main__':


	# 加载训练好的模型
	model_filename = 'random_forest_model.pkl'
	with open('/workspace/codes/toxicity/rf_maccs.pkl', 'rb') as model_file:
		loaded_model = pickle.load(model_file)

	# 准备新的数据
	with open('/workspace/code/200_gsf0_200_simple_tox_22_rescale.pkl','rb')as f:
		new_data = pickle.load(f)
	mmm = []
	molecules = [Chem.MolFromSmiles(smiles) for smiles in new_data]
	maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules]
	fp_array = np.array([list(fp) for fp in maccs_fps])
	# for i in new_data:
	# 	mmm.append(np.array(smiles_to_one_hot(i)).flatten())
	# 使用加载的模型进行预测
	new_predictions = loaded_model.predict(fp_array)
	cnt=0
	for i in new_predictions:
		if i ==1 :
			cnt +=1
	# 输出预测结果
	print(new_predictions)