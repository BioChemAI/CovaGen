"""https://stackoverflow.com/a/61395551/16407115
"""

from io import StringIO
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select


def is_het(residue):
    res = residue.id[0]
    return res != " " and res != "W"


class ResidueSelect(Select):
    def __init__(self, chain, residue):
        self.chain = chain
        self.residue = residue

    def accept_chain(self, chain):
        return chain.id == self.chain.id

    def accept_residue(self, residue):
        """ Recognition of heteroatoms - Remove water molecules """
        return residue == self.residue and is_het(residue)


def extract_ligands(pfb_file: str):
    """ Extraction of the heteroatoms of .pdb files """

    pfb_file = Path(pfb_file)
    pdb_code = pfb_file.with_suffix("").name
    pdb = PDBParser().get_structure(pdb_code, str(pfb_file))
    io = PDBIO()
    io.set_structure(pdb)
    for model in pdb:
        for chain in model:
            for residue in chain:
                if not is_het(residue):
                    continue
                f = StringIO()
                print(f"Found {chain} {residue}")
                io.save(f, ResidueSelect(chain, residue))
                pdb_content = f.getvalue()
                f.close()
                yield pdb_content


class ResidueSelect2(Select):
    def __init__(self):
        pass

    def accept_chain(self, chain):
        return 1

    def accept_residue(self, residue):
        """ Recognition of heteroatoms - Remove water molecules """
        res = residue.id[0]
        return res == " "

def extract_protein(pfb_file: str):
    """ Extraction of the heteroatoms of .pdb files """

    pfb_file = Path(pfb_file)
    pdb_code = pfb_file.with_suffix("").name
    pdb = PDBParser().get_structure(pdb_code, str(pfb_file))
    io = PDBIO()
    io.set_structure(pdb)

    f = StringIO()
    io.save(f, ResidueSelect2())
    pdb_content = f.getvalue()
    f.close()
    return pdb_content

if __name__ == '__main__':
    path = "../save_2/all_in_one/download/2xba.pdb"
    # for c in extract_ligands(path):
    #     print(c)

    c = extract_protein(path)
    print(c)
    with open("../tmp/2xba_extracted.pdb", "w") as f:
        f.write(c)