import os,sys
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm
sys.path.append("../")
from transvae.trans_models import TransVAE
from transvae.tvae_util import decode_mols
from torch.autograd import Variable
from torch_scatter import scatter_add,scatter_mean
from transvae.tvae_util import encode_smiles,tokenizer




class PDBProtein(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.atom_dic = ['C', 'N', 'H', 'S', 'O','unk']
        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms

        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)

            # atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            # self.element.append(atomic_number)
            try:

                self.element.append(self.atom_dic.index(atom['element_symb']))
            except:
                print('unk atom appear: {}'.format(atom['element_symb']))
                self.element.append(self.atom_dic.index('unk'))


            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            try:
                self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])
            except:
                self.atom_to_aa_type.append(20)
                print(atom['res_name'])

    def to_dict_atom(self):
        return {
            'pos': torch.tensor(self.pos, dtype=torch.float32),
            'atom': torch.tensor(self.element, dtype=torch.long),
            'is_backbone': torch.tensor(self.is_backbone, dtype=torch.bool),
            'res': torch.tensor(self.atom_to_aa_type, dtype=torch.long)
        }
def smiles_to_embed(smiles,vae_model):
    try:
        smi_token = tokenizer(smiles)
        encoded_smi = [0] + encode_smiles(smi_token, 126, vae_model.params['CHAR_DICT']) #127
    except Exception as e:
        print("encode exception:",e)
        return None
    encoded_smi = torch.tensor(encoded_smi).unsqueeze(0)
    if vae_model.use_gpu:
        encoded_smi = encoded_smi.cuda()
    src = Variable(encoded_smi).long()
    src_mask = (src != vae_model.pad_idx).unsqueeze(-2)
    # mem, mu, logvar, _, mem_z = vae_model.model.encode(src, src_mask)
    mem, mu, logvar = vae_model.model.encode(src)
    return mu.detach().cpu().squeeze()









if __name__=='__main__':
    import pandas as pd
    data_dir='/mnt/lpy/xingzhi_competition/data_7_4_hashed/'
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

    system_ids = train_df['System_ID']
    complex_ids = train_df['Complex_ID']

    system_id = system_ids[12750]
    conformer_id = complex_ids[12750]

    complex_path = os.path.join(data_dir, 'conformers', system_id, 'conformers', conformer_id,
                                'complex_out.pdb')
    complex_dict = PDBProtein(complex_path).to_dict_atom()
