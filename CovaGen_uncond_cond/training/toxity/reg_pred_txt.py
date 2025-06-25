import pickle

import pandas as pd
import xgboost as xgb
from rdkit import  Chem
from rdkit.Chem import  AllChem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open('./xgboost_model_maccs.pkl', "rb") as model_file:
    loaded_model = pickle.load(model_file)
with open('/workspace/codes/Docking/ddpo_todock/fulldata_s0.pkl','rb') as f:
    # zero_list  = [i.rstrip() for i in f.readlines()]
    zero_list = pickle.load(f)
with open('/workspace/codes/Docking/ddpo_todock/1725_todock.pkl','rb') as f:
    # cleaned_list  = [i.rstrip() for i in f.readlines()]
    cleaned_list = pickle.load(f)
with open('/workspace/code/200_gs0_200_tox_pmean_rescale_model24000.pkl','rb') as f:
    zero_list_24000  = pickle.load(f)#

# crossdock_train = pd.read_csv('/workspace/code/crossdock_train.csv')
# cross = list(set(crossdock_train['mol'].tolist()))
cleaned_list = list(set(cleaned_list))

zero_mols = [Chem.MolFromSmiles(smiles) for smiles in zero_list]
molecules = [Chem.MolFromSmiles(smiles) for smiles in cleaned_list]
zero_mols_24000 = [Chem.MolFromSmiles(smiles) for smiles in zero_list_24000]
	#
	# 计算MACCS指纹
zero_maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in zero_mols if mol is not None]
maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
zero_maccs_fps_24000 = [AllChem.GetMACCSKeysFingerprint(mol) for mol in zero_mols_24000 if mol is not None]


zero_fp_array = np.array([list(fp) for fp in zero_maccs_fps])
fp_array = np.array([list(fp) for fp in maccs_fps])
zero_fp_array_24000 = np.array([list(fp) for fp in zero_maccs_fps_24000])

# 使用加载的模型进行预测

dtest = xgb.DMatrix(fp_array)
zero_dtest = xgb.DMatrix(zero_fp_array)
zero_dtest_24000 = xgb.DMatrix(zero_fp_array_24000)
y_pred_loaded = loaded_model.predict(dtest)
y_pred_loaded_zero = loaded_model.predict(zero_dtest)
y_pred_loaded_zero_24000 = loaded_model.predict(zero_dtest_24000)

# y_pred_loaded = loaded_model.predict(fp_array)
# y_pred_loaded_zero = loaded_model.predict(zero_fp_array)
# y_pred_loaded_zero_24000 = loaded_model.predict(zero_fp_array_24000)
sns.set(style="whitegrid")
sns.kdeplot(y_pred_loaded, label = "data guided",shade=True)
sns.kdeplot(y_pred_loaded_zero, label="Data zero", shade=True)
# sns.kdeplot(y_pred_loaded_zero_24000, label="Data zero_24000", shade=True)
# 添加标题和标签
plt.title("Density Distribution Comparison")
plt.xlabel("Values")
plt.ylabel("Density")
# 添加图例
plt.legend()

# 显示图形
plt.show()

# 显示图形
plt.show()
print('done')