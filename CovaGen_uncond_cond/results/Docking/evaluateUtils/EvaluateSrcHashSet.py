import hashlib
from .to_mol import mol_to_smiles, smiles_to_mol, sdf_to_mol

class EvaluateSrcHashSet():
    def __init__(self) -> None:
        self._list = []
        self._data = None

    def __len__(self):
        return len(self._list)

    def __getitem__(self, idx):
        return self._list[idx]

    def __setitem__(self, idx, data):
        self._list[idx] = data

    def hash_many(self, *args):
        m = hashlib.sha256()
        for data in args:
            if isinstance(data, str):
                data = data.encode()
            m.update(hashlib.sha256(data).digest())
        return m.hexdigest()

    def update(self, _from, receptor_pdb, ligand_smiles=None, ligand_sdf=None, ligand_mol=None, box=None, more_kwargs=dict()):
        if sum((bool(ligand_smiles), bool(ligand_sdf))) != 1:
            return None

        if ligand_smiles:
            _hash = self.hash_many(receptor_pdb, ligand_smiles, str(box))
            evaluate_src = next((x for x in self._list if x["_hash"] == _hash), None)
            if evaluate_src:
                evaluate_src["_from_list"].append(_from)
            else:
                if not ligand_mol:
                    ligand_mol = smiles_to_mol(ligand_smiles) 
                evaluate_src = {
                    "receptor_pdb": receptor_pdb,
                    "ligand_mol": ligand_mol,
                    "ligand_smiles": ligand_smiles,
                    "box": box,
                    "more_kwargs": more_kwargs,
                    "_hash": _hash,
                    "_from_list": [_from]
                }
                self._list.append(evaluate_src)

        if ligand_sdf:
            _hash = self.hash_many(receptor_pdb, ligand_sdf, str(box))
            evaluate_src = next((x for x in self._list if x["_hash"] == _hash), None)
            if evaluate_src:
                evaluate_src["_from_list"].append(_from)
            else:
                if not ligand_mol:
                    ligand_mol = sdf_to_mol(ligand_sdf)
                evaluate_src = {
                    "receptor_pdb": receptor_pdb,
                    "ligand_mol": ligand_mol,
                    "ligand_smiles": mol_to_smiles(ligand_mol),
                    "box": box,
                    "more_kwargs": more_kwargs,
                    "_hash": _hash,
                    "_from_list": [_from]
                }
                self._list.append(evaluate_src)

        return evaluate_src

    def check(self):
        length_before = len(self._list)
        self._list = list(filter(lambda x: x["ligand_mol"] is not None, self._list))
        length_after = len(self._list)
        return length_before, length_after
