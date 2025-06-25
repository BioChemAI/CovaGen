import pickle

import numpy as np
import pandas as pd
# from Bio import PDB
# from Bio.SeqUtils import seq3
import torch
# from Bio.PDB.Polypeptide import PPBuilder
from rdkit import Chem
from tqdm.auto import tqdm

class residue_constants:
    restype_1to3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }

    # restype_3to1 = {v: k for k, v in restype_1to3.items()}
    restype_3to1 ={
    'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
    'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
    'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
    'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',
    '2AS':'D', '3AH':'H', '5HP':'E', 'ACL':'R', 'AIB':'A',
    'ALM':'A', 'ALO':'T', 'ALY':'K', 'ARM':'R', 'ASA':'D',
    'ASB':'D', 'ASK':'D', 'ASL':'D', 'ASQ':'D', 'AYA':'A',
    'BCS':'C', 'BHD':'D', 'BMT':'T', 'BNN':'A', 'BUC':'C',
    'BUG':'L', 'C5C':'C', 'C6C':'C', 'CCS':'C', 'CEA':'C',
    'CHG':'A', 'CLE':'L', 'CME':'C', 'CSD':'A', 'CSO':'C',
    'CSP':'C', 'CSS':'C', 'CSW':'C', 'CXM':'M', 'CY1':'C',
    'CY3':'C', 'CYG':'C', 'CYM':'C', 'CYQ':'C', 'DAH':'F',
    'DAL':'A', 'DAR':'R', 'DAS':'D', 'DCY':'C', 'DGL':'E',
    'DGN':'Q', 'DHA':'A', 'DHI':'H', 'DIL':'I', 'DIV':'V',
    'DLE':'L', 'DLY':'K', 'DNP':'A', 'DPN':'F', 'DPR':'P',
    'DSN':'S', 'DSP':'D', 'DTH':'T', 'DTR':'W', 'DTY':'Y',
    'DVA':'V', 'EFC':'C', 'FLA':'A', 'FME':'M', 'GGL':'E',
    'GLZ':'G', 'GMA':'E', 'GSC':'G', 'HAC':'A', 'HAR':'R',
    'HIC':'H', 'HIP':'H', 'HMR':'R', 'HPQ':'F', 'HTR':'W',
    'HYP':'P', 'IIL':'I', 'IYR':'Y', 'KCX':'K', 'LLP':'K',
    'LLY':'K', 'LTR':'W', 'LYM':'K', 'LYZ':'K', 'MAA':'A',
    'MEN':'N', 'MHS':'H', 'MIS':'S', 'MLE':'L', 'MPQ':'G',
    'MSA':'G', 'MSE':'M', 'MVA':'V', 'NEM':'H', 'NEP':'H',
    'NLE':'L', 'NLN':'L', 'NLP':'L', 'NMC':'G', 'OAS':'S',
    'OCS':'C', 'OMT':'M', 'PAQ':'Y', 'PCA':'E', 'PEC':'C',
    'PHI':'F', 'PHL':'F', 'PR3':'C', 'PRR':'A', 'PTR':'Y',
    'SAC':'S', 'SAR':'G', 'SCH':'C', 'SCS':'C', 'SCY':'C',
    'SEL':'S', 'SEP':'S', 'SET':'S', 'SHC':'C', 'SHR':'K',
    'SOC':'C', 'STY':'Y', 'SVA':'S', 'TIH':'A', 'TPL':'W',
    'TPO':'T', 'TPQ':'A', 'TRG':'K', 'TRO':'W', 'TYB':'Y',
    'TYQ':'Y', 'TYS':'Y', 'TYY':'Y', 'AGM':'R', 'GL3':'G',
    'SMC':'C', 'ASX':'B', 'CGU':'E', 'CSX':'C', 'GLX':'Z',
    'PYX':'C',
    'UNK':'X'
    }



def chain_to_res3d(chain):
    """Convert chain into sequence and atomic point cloud.
    Refer to the method used by esm.

    https://github.com/aqlaboratory/openfold/blob/main/openfold/np/protein.py
    """
    seq: str = ''


    for res in chain:

        # Refer to this link. The alphabet of esm supports 'X' for unknow residue.
        # https://github.com/facebookresearch/esm/blob/main/esm/constants.py
        resname = res.get_resname()
        res_shortname = residue_constants.restype_3to1.get(resname, "X")
        seq += res_shortname

    return seq

# 创建PDB解析器
# parser = PDB.PDBParser(QUIET=True)

with open("/workspace/datas/crossdock_protein/crossdocked_pocket10/index.pkl",'rb')as f:
    ind  = pickle.load(f)

spl = torch.load('/workspace/datas/crossdock_protein/crossdocked_pocket10/split_by_name.pt')

seq_ls = []
ress ={}
# 读取PDB文件
# structure = parser.get_structure("protein", "/workspace/datas/crossdock_protein/crossdocked_pocket10/1433B_HUMAN_1_240_pep_0/5f74_A_rec.pdb")
tes = spl['test']
train_root = '/workspace/datas/crossdock_protein/crossdocked_pocket10/'
df = pd.DataFrame(columns=['id', 'smiles', 'pocket sequence', 'protein sequence'])
pbar = tqdm(tes)
for i in pbar:
    res_lst = []
    fname = i[0]
    part1 = fname.split('/')[0]
    part2 = fname.split('/')[1]
    parts = part2.split('_')

    chain_ind = parts[1]
    # chain_ind = int(''.join(str(ord(char) - ord('A') + 1) for char in chain_ind))-1
    # 提取第三个下划线之前的部分
    result = part1+'/'+'_'.join(parts[:3])+'.pdb'

    pfname = '/workspace/datas/crossdock_protein/crossdocked_pocket10/'+result
    pocket_fname = train_root + i[0]
    ligand_fname = train_root + i[1]

    # The protein chain
    protein_sequence = ""
    structure = parser.get_structure("protein",pfname)
    model = structure[0]
    chains = list(model)
    # for chain in chains:
    for i in chains:
        if i.id == chain_ind:
            aa = chain_to_res3d(i)
            break

    # The pocket sequence
    pocket_sequence = ""
    structure_pocket = parser.get_structure('protein',pocket_fname)
    model_pocket = structure[0]
    chains_pocket = list(model)
    for i in chains:
        aa_pocket = chain_to_res3d(i)
    # ppb = PPBuilder()
    # ppstruct = ppb.build_peptides(structure[0])
    # print('done')
    # for model in structure:
    #     for chain in model:
    #         if chain.id.isalnum() and chain.id.strip():  # 检查是否为蛋白质链
    #             for residue in chain:
    #                 if PDB.is_aa(residue):  # 检查是否为氨基酸
    #                     protein_sequence += three_to_one(residue.get_resname())\
    supplier = Chem.SDMolSupplier(ligand_fname)

    # 提取每个分子的 SMILES
    smiles_list = []
    for mol in supplier:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
    res_lst.append(result)
    res_lst.extend(smiles_list)
    res_lst.append(aa)
    res_lst.append(aa_pocket)
    df.loc[len(df)] = res_lst
df.to_csv("/workspace/codes/cross_docked_seqs_test.csv")
    # ress.update({result: aa})
    # seq_ls.append(aa)
# model = structure[0]
# chains = list(model)
# for chain in chains:
#     aa = chain_to_res3d(chain)

# for model in structure:
#     for chain in model:
#         if chain.id.isalnum() and chain.id.strip():  # 检查是否为蛋白质链
#             for residue in chain:
#                 if PDB.is_aa(residue):  # 检查是否为氨基酸
#                     protein_sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
# with open('/workspace/codes/testset_seqs.pkl','wb')as f:
#     pickle.dump(ress,f)
# print("Protein Sequence:", protein_sequence)
# print(len(protein_sequence))
print('done')