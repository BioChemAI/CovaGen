from itertools import combinations
from typing import List
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.rdBase import BlockLogs
import numpy as np

def smiles_to_scaffold_fp(smiles: str):
    block = BlockLogs()
    m = Chem.MolFromSmiles(smiles)
    m = MurckoScaffold.GetScaffoldForMol(m) if m else None
    del block
    return Chem.RDKFingerprint(m) if m else None

def smiles_list_to_scaffold_fps(smiles_list: str, clean=True):
    fps = [smiles_to_scaffold_fp(x) for x in smiles_list]
    return [x for x in fps if x] if clean else fps

def smiles_to_fp(smiles: str):
    block = BlockLogs()
    m = Chem.MolFromSmiles(smiles)
    del block
    return Chem.RDKFingerprint(m) if m else None

def smiles_list_to_fps(smiles_list: str, clean=True):
    fps = [smiles_to_fp(x) for x in smiles_list]
    return [x for x in fps if x] if clean else fps

def get_similarity_comb(fps: List):
    similarity_list = [
        DataStructs.FingerprintSimilarity(fp_x, fp_y) \
            for fp_x, fp_y in combinations(fps, 2)
    ]
    return np.mean(similarity_list)

def get_similarity_ref_prd(ref_fp, prd_fps: List[str]):
    similarity_list = [
        DataStructs.FingerprintSimilarity(ref_fp, prd_fp) \
            for prd_fp in prd_fps
    ]
    return np.mean(similarity_list)
    
def get_novelty_train_prd(train_fps: List, prd_fps: List[str], threshold=0.4):
    count = 0
    if len(prd_fps)==0:
        return 0
    for fp in prd_fps:
        for train_fp in train_fps:
            similarity = DataStructs.FingerprintSimilarity(train_fp, fp)
            if similarity > threshold:
                count += 1
                break
    return 1 - count / len(prd_fps)

def get_novelty_train_prd_smiles(train_smiles_lst: List, prd_smiles_lst: List[str]):
    prd_smiles_lst = [x for x in prd_smiles_lst if x]
    if len(prd_smiles_lst)==0:
        return 0
    novel_smiles_set = set(prd_smiles_lst) - set(train_smiles_lst)
    count = sum((x in novel_smiles_set) for x in prd_smiles_lst)
    return count / len(prd_smiles_lst)
    
