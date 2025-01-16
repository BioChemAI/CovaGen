"""
Estimate Novelty & Diversity

## Usage

```bash
model_type=egnnCvaeProteinToLigand
train_id=20221115_044949_5fc8_sbdd
postfix=_p0.7_u100

python -m debugpy --listen 5678 --wait-for-client evaluate.novelty_diversity.py \
    --collect-pkl saved/$model_type/$train_id/evaluate/collect$postfix.pkl \
    --index-json saved/preprocess/index_sbdd.json \
    --frag-ligand-pkl saved/preprocess/frag_ligand_sbdd.pkl \
    --output-json saved/$model_type/$train_id/evaluate/novelty_diversity$postfix.json

python evaluate.novelty_diversity.py \
    --collect-pkl saved/$model_type/$train_id/evaluate/collect$postfix.pkl \
    --index-json saved/preprocess/index_sbdd.json \
    --frag-ligand-pkl saved/preprocess/frag_ligand_sbdd.pkl \
    --output-json saved/$model_type/$train_id/evaluate/novelty_diversity$postfix.json
```
"""

import argparse
import logging
import pickle
import json
from tqdm import tqdm
from evaluateUtils.distribution import get_novelty_train_prd, get_novelty_train_prd_smiles, get_similarity_comb, get_similarity_ref_prd, smiles_list_to_fps, smiles_list_to_scaffold_fps, smiles_to_fp, smiles_to_scaffold_fp

from utils.common import init_logging, use_path
from evaluateUtils.EvaluateSrcHashSet import EvaluateSrcHashSet

def novelty_diversity_sin(evaluate_src_hash_set: EvaluateSrcHashSet, index_list,
        frag_ligand_list):

    sample_smiles_list, ref_smiles_list = evaluate_src_hash_set._data

    train_code_list = [x["code"] for x in index_list if x["split"] == "train"]
    train_code_set = set(train_code_list)
    train_smiles_list = [x["smiles"] for x in frag_ligand_list if x["code"] in train_code_set]
    train_smiles_list = [x for x in train_smiles_list if x]
    train_smiles_list = list(set(train_smiles_list))

    logging.info("Get fingerprints in train set...")
    train_fp_list = smiles_list_to_fps(train_smiles_list)
    logging.info(f"Done! {len(train_smiles_list)} > {len(train_fp_list)}")
    logging.info("Get scaffold fingerprints in train set...")
    train_scaffold_fp_list = smiles_list_to_scaffold_fps(train_smiles_list)
    logging.info(f"Done! {len(train_smiles_list)} > {len(train_scaffold_fp_list)}")

    result_dict = dict()

    pbar = tqdm(list(zip(sample_smiles_list, ref_smiles_list)), desc="Evaluate [novelty_diversity]")

    for sample_smiles_item, ref_smiles in pbar:
        code = sample_smiles_item["code"]
        prd_smiles_list = sample_smiles_item["smiles"]

        prd_fp_list = smiles_list_to_fps(prd_smiles_list)
        prd_scaffold_fp_list = smiles_list_to_scaffold_fps(prd_smiles_list)
        
        ref_fp = smiles_to_fp(ref_smiles)
        ref_scaffold_fp = smiles_to_scaffold_fp(ref_smiles)

        result = dict(
            similarity_prd_comb = get_similarity_comb(fps=prd_fp_list),
            similarity_ref_prd = get_similarity_ref_prd(ref_fp=ref_fp, prd_fps=prd_fp_list),
            novelty_train_prd = get_novelty_train_prd(train_fps=train_fp_list, prd_fps=prd_fp_list),
            novelty_train_prd_smiles = get_novelty_train_prd_smiles(train_smiles_lst=train_smiles_list, prd_smiles_lst=prd_smiles_list),
            scaffold = dict(
                similarity_prd_comb = get_similarity_comb(fps=prd_scaffold_fp_list),
                similarity_ref_prd = get_similarity_ref_prd(ref_fp=ref_scaffold_fp, prd_fps=prd_scaffold_fp_list),
                novelty_train_prd = get_novelty_train_prd(train_fps=train_scaffold_fp_list, prd_fps=prd_scaffold_fp_list)
            )
        )

        result_dict[code] = result
        pbar.set_postfix({k:v for k, v in result.items() if k not in ("scaffold", )})

    return result_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Estimate Novelty & Diversity.')
    parser.add_argument("--index-json", type=str)
    parser.add_argument("--collect-pkl", type=str)
    parser.add_argument("--frag-ligand-pkl", type=str)
    parser.add_argument("--output-json", type=str)
    args = parser.parse_args()

    collect_fn = use_path(file_path=args.collect_pkl, new=False)

    output_fn = use_path(file_path=args.output_json)
    log_fn = use_path(file_path=f"{args.output_json}.log")
    
    init_logging(log_fn)
    logging.info(args)

    with open(collect_fn, 'rb') as f:
        evaluate_src_hash_set: EvaluateSrcHashSet = pickle.load(f)
    logging.info(f"Loaded! evaluate_src_hash_set: {collect_fn}")

    with open(args.index_json, 'r') as f:
        index_list = json.load(f)

    with open(args.frag_ligand_pkl, 'rb') as f:
        frag_ligand_list = pickle.load(f)

    logging.info("Estimate Novelty & Diversity...")

    result_dict = novelty_diversity_sin(
        evaluate_src_hash_set, 
        index_list,
        frag_ligand_list
    )

    with open(output_fn, 'w') as f:
        json.dump(result_dict, f, indent=4)
    logging.info(f"Saved! result_dict: {output_fn}")

    logging.info("Finish!")
    