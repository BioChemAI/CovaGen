import pickle

import pandas as pd
import xgboost as xgb
from rdkit import  Chem
from rdkit.Chem import  AllChem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import random
from sklearn.manifold import TSNE
# with open("/workspace/codes/toxicity/acute_with_dimension.csv")

k = pd.read_csv("/workspace/codes/toxicity/acute_with_dimension.csv")
k=k.dropna()
list1 = []
list2 = []
for index, row in k.iterrows():
    if row["Toxicity Value"] > 5000:
        list1.append(row["Canonical SMILES"])
    if row["Toxicity Value"] < 5:
        list2.append(row["Canonical SMILES"])
list1 = random.sample(list1,99)
list2 = random.sample(list2,99)

molecules1 = [Chem.MolFromSmiles(smiles) for smiles in list1]
	#
	# 计算MACCS指纹
# maccs_fps1 = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules1]
maccs_fps1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in molecules1]
fp_array1 = np.array([list(fp) for fp in maccs_fps1])

molecules2 = [Chem.MolFromSmiles(smiles) for smiles in list2]
	#
	# 计算MACCS指纹
# maccs_fps2 = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules2]
maccs_fps2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in molecules2]
fp_array2 = np.array([list(fp) for fp in maccs_fps2])
data = np.concatenate((fp_array1,fp_array2))
# reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2)
# embedding = reducer.fit_transform(data)
tsne = TSNE(n_components=2, random_state=42)  # 选择要降维到的目标维度，这里选择2维
embedding = tsne.fit_transform(data)
list1_embedding = embedding[:len(list1)]
list2_embedding = embedding[len(list1):]
plt.scatter(list1_embedding[:, 0], list1_embedding[:, 1], label='list1', c='blue')
plt.scatter(list2_embedding[:, 0], list2_embedding[:, 1], label='list2', c='red')
plt.legend()
plt.title('UMAP Visualization')
plt.show()
print("done")