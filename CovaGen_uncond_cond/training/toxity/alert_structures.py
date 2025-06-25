from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
# 定义SMARTS表示的子结构
# substructure_smarts = ['[ClX1][CX4]', '[NX1]#[CX2]','[NX3H0+0,NX4H1+;!$([N][!C]);!$([N]*~[#7,#8,#15,#16])]','[a;!c]','[nX2,nX3+]']

substructure_smarts = ['Cc1c(c(ccc1)C(C))', ' C(C(COC)CC)','c1c(c(c(c(c1Cl)Cl))Cl)Cl','C(CO)Cl','c1cc2ccccc2cc1','c1c(c(cc(c1)CC))[OH]','c1cc(ccc1Cc2ccc(cc2))','c1ccc(cc1)c2ccccc2'] # Liver toxicity from Identification of structural alerts for liver and kidney toxicity using repeated dose toxicity data
# 从文件中读取分子数据，假设存储为SMILES格式
with open("/workspace/codes/othercodes/lddd_substructures/egfr_substructure/s10000_pos_ddpo_egfr_1575.pkl",'rb')as f:
    molecules_smiles = pickle.load(f)[0:1000]

# 初始化计数器
substructure_count = {smarts: 0 for smarts in substructure_smarts}

# 遍历分子列表并统计子结构出现次数#
for smiles in molecules_smiles:#
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for smarts in substructure_smarts:
            pattern = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pattern):
                substructure_count[smarts] += 1
cnt = 0
# 打印结果
for smarts, count in substructure_count.items():
    print(f'Substructure {smarts} appears {count} times.')
    cnt+=count
print('Total appearance:', cnt)