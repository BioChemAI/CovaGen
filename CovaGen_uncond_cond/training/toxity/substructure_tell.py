import pickle

from rdkit import Chem

def has_benzene_ring(molecule):
    try:
        mol = Chem.MolFromSmiles(molecule)  # 从 SMILES 格式创建分子对象
        if mol is None:
            return False

        # 判断分子是否具有苯环
        ssr = Chem.GetSymmSSSR(mol)  # 获取分子的最小环系统
        for ring in ssr:
            if len(ring) == 6:  # 判断是否为6个原子的环
                bond_types = [mol.GetBondBetweenAtoms(ring[i], ring[i+1]).GetBondType() for i in range(5)]
                if all(bt == Chem.BondType.AROMATIC for bt in bond_types):
                    return True

        return False
    except:
        return False
if __name__ == '__main__':

	# 测试分子 SMILES
	molecule_smiles = "c1ccccc1"  # 苯环的 SMILES 表示
	with open('/workspace/code/200_gs10_1000_tox.pkl','rb')as f:
		k = pickle.load(f)
	cnt = 0
	# 判断分子是否具有苯环
	for i in k:
		if has_benzene_ring(i):
			cnt += 1
	print('done')
	result = has_benzene_ring(molecule_smiles)