import pickle

from rdkit import Chem

# 创建一个分子对象，可以从SMILES字符串或其他方式创建
with open('/workspace/code/200_gs0_ddpo_toxrescale_allpok5.pkl','rb') as f:
    cleaned_list  = pickle.load(f)
with open('/workspace/code/200_gs0_200_tox_pmean_rescale_model24000.pkl','rb') as f:
    zero_list_24000  = pickle.load(f)#
cleaned_list = list(set(cleaned_list))
mol1 = [Chem.MolFromSmiles(mol) for mol in cleaned_list]
mol2 = [Chem.MolFromSmiles(mol) for mol in zero_list_24000]
cnt = 0
# 检测分子中是否含有磷原子
for i in mol1:
    has_phosphorus = any(atom.GetSymbol() == "P" for atom in i.GetAtoms())
    if has_phosphorus:
        cnt+=1
    else:
        cnt+=0
print('done')