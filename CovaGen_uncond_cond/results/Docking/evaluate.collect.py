"""
Collect the data required for evaluation (docking, qed, sa, etc.) 
Hash the evaluation content to avoid repeated testing

## Usage

```bash
model_type=cvaeProteinToLigand
train_id=20221026_185659_c2b0_sbdd

python evaluate.collect.py \
    --sample-smiles-json saved/$model_type/$train_id/sample/smiles.json \
    --index-json saved/preprocess/index_sbdd.json \
    --data-dir $sbdd_dir \
    --receptor protein \
    --receptor-box protein \
    --ref-from-smiles \
    --ref-from-sdf \
    --collect-pkl saved/$model_type/$train_id/evaluate/collect.pkl
```

```bash
model_type=egnnCvaeProteinToLigand
train_id=20221102_170634_f481_sbdd

python evaluate.collect.py \
    --sample-smiles-json saved/$model_type/$train_id/sample/smiles.json \
    --index-json saved/preprocess/index_sbdd.json \
    --data-dir $sbdd_dir \
    --receptor protein \
    --receptor-box pocket \
    --ref-from-smiles \
    --ref-from-sdf \
    --collect-pkl saved/$model_type/$train_id/evaluate/collect.pkl
```

"""

import argparse
import json
import logging
import pickle
from tqdm import tqdm

from evaluateUtils.get_box import get_box
from evaluateUtils.to_mol import sdf_to_smiles
from evaluateUtils.EvaluateSrcHashSet import EvaluateSrcHashSet
from utils.common import find_by_key_value_cached, init_logging, use_path


if __name__ == '__main__':##

    parser = argparse.ArgumentParser(description='Docking.')
    parser.add_argument("--sample-smiles-json", type=str)
    parser.add_argument("--index-json", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--receptor", type=str, choices=["pocket", "protein"])
    parser.add_argument("--receptor-box", type=str, choices=["pocket", "protein"])
    parser.add_argument("--ref-from-smiles", action="store_true")
    parser.add_argument("--ref-from-sdf", action="store_true")
    parser.add_argument("--collect-pkl", type=str)
    parser.add_argument("--index-protein", type=str)
    args = parser.parse_args()

    # Define Path
    sample_smiles_fn = use_path(file_path=args.sample_smiles_json, new=False)
    index_fn = use_path(file_path=args.index_json, new=False)
    data_dn = use_path(dir_path=args.data_dir, new=False)

    collect_fn = use_path(file_path=args.collect_pkl)
    log_fn = use_path(file_path=f"{args.collect_pkl}.log")

    init_logging(log_fn)
    logging.info(args)

    with open(sample_smiles_fn, 'r') as f:
        sample_smiles_list = json.load(f)

    # with open(args.index_json, 'r') as f:
    #     index_list = json.load(f)
    with open(args.index_protein, "r") as f:
        index_things = pickle.load(f)
    evaluate_src_hash_set = EvaluateSrcHashSet()

    logging.info("Collect evaluate src...")
    # find_code_in_index_list = find_by_key_value_cached(index_list, "code")

    ref_smiles_list = []
    sample_smiles_pbar = tqdm(sample_smiles_list, desc="Evaluate [collect]")
    for sample_smiles in sample_smiles_pbar:

        code = sample_smiles["code"]
        # index_item = find_code_in_index_list(code)
        dir_name = sample_smiles["dirfirst"]
        # protein_pocket_name = index_path.split(".")[0][:-9]
        nm = code.split("_")
        nm1 = nm[:3]
        protein_name = '_'.join(nm1)
        protein_fl = dir_name + "/" + protein_name + ".pdb"
        ligand_fl = dir_name + "/" + code + ".sdf"
        pocket_fl = dir_name + "/" + code + "_pocket10" + ".pdb"

        with open(data_dn / protein_fl, 'rb') as f:
            receptor_pdb = f.read()
        with open(data_dn / ligand_fl, 'rb') as f:
            ref_ligand_sdf = f.read()
#
        ref_ligand_smiles = sdf_to_smiles(ref_ligand_sdf)
        ref_smiles_list.append(ref_ligand_smiles)

        if args.receptor_box == "protein":
            center, size = get_box(receptor_pdb=receptor_pdb)
        else:
            center, size = get_box(ligand_sdf=ref_ligand_sdf)#

        if args.ref_from_smiles:
            evaluate_src_hash_set.update(
                (code, "ref_smiles", 0), 
                receptor_pdb = receptor_pdb, 
                ligand_smiles = ref_ligand_smiles,
                box = (center, size)
            )

        if args.ref_from_sdf:
            evaluate_src_hash_set.update(
                (code, "ref_sdf", 0), 
                receptor_pdb = receptor_pdb, 
                ligand_sdf = ref_ligand_sdf,
                box = (center, size)
            )

        for prd_idx, prd_smiles in enumerate(sample_smiles["smiles"]):
            if prd_smiles and isinstance(prd_smiles, str):
                evaluate_src_hash_set.update(
                    (code, "prd_smiles", prd_idx), 
                    receptor_pdb = receptor_pdb, 
                    ligand_smiles = prd_smiles,
                    box = (center, size)
                )

    length_before, length_after = evaluate_src_hash_set.check()
    logging.info(f"Done! evaluate_src_hash_set, length_before: {length_before}, length_now: {length_after}")

    # In this way, you don't need to read it separately during analysis
    evaluate_src_hash_set._data = (sample_smiles_list, ref_smiles_list)

    with open(args.collect_pkl, 'wb') as f:
        pickle.dump(evaluate_src_hash_set, f)
    logging.info(f"Saved! collect_pkl: {args.collect_pkl}")

    logging.info("Finish!")
    
