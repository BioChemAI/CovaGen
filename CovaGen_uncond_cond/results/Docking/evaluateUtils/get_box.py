import tempfile

import numpy as np
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from .to_mol import sdf_to_mol

def get_box(ligand_sdf = None, ref_mol = None, receptor_pdb = None):
    block = BlockLogs()

    if ligand_sdf is not None:
        ref_mol = sdf_to_mol(ligand_sdf)
    # a = [3.26,4.0,15.23]
    # b= [18,18,18]
    # center = np.array(a)
    # size = np.array(b)
    # return center,size

    if ref_mol is not None:
        pos = ref_mol.GetConformer(0).GetPositions()
        center = (pos.max(0) + pos.min(0)) / 2
        size = pos.max(0) - pos.min(0) + 5
        size[size < 20] = 20
        return center, size

    if receptor_pdb is not None:
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(receptor_pdb)
            mol = Chem.MolFromPDBFile(fp.name, sanitize=False)
        pos = mol.GetConformer(0).GetPositions()
        center = (pos.max(0) + pos.min(0)) / 2
        size = pos.max(0) - pos.min(0) + 10
        return center, size

    del block
    raise Exception("No input")

if __name__ == '__main__':
    with open('/workspace/codes/Docking/singles/pick2/Structures.pdb', 'rb') as f:
        receptor_pdb = f.read()
    c,s = get_box(receptor_pdb=receptor_pdb)
    print('done')