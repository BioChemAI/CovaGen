"""
Collect smiles database.

## Usage

```bash
model_type=cvaeProteinToLigand
train_id=20221115_105154_1b96_sbdd
postfix=_u100

python evaluate.collect.smiles_db.py \
    --collect-pkl saved/$model_type/$train_id/evaluate/collect$postfix.pkl \
    --db-json saved/common/smiles_db.json
```
"""

import argparse
from itertools import chain
import logging
import pickle
import json

from utils.common import init_logging, use_path
from evaluateUtils.EvaluateSrcHashSet import EvaluateSrcHashSet

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Collect smiles database')
    parser.add_argument("--collect-pkl", type=str)
    parser.add_argument("--db-json", type=str)
    args = parser.parse_args()

    collect_fn = use_path(file_path=args.collect_pkl, new=False)

    db_fn = use_path(file_path=args.db_json)
    log_fn = use_path(file_path=f"{args.db_json}.log")
    
    init_logging(log_fn)
    logging.info(args)

    with open(collect_fn, 'rb') as f:
        evaluate_src_hash_set: EvaluateSrcHashSet = pickle.load(f)
    logging.info(f"Loaded! evaluate_src_hash_set: {collect_fn}")

    logging.info(f"New job!")
    logging.info("Collect smiles database...")

    sample_smiles_list, ref_smiles_list = evaluate_src_hash_set._data#
    smiles_list_with_none = list(chain(*(x["smiles"] for x in sample_smiles_list), ref_smiles_list))
    smiles_list = list(x for x in smiles_list_with_none if x)
    # smiles_list = sample_smiles_list
    smiles_set = set(smiles_list)
    
    if db_fn.exists():
        with open(db_fn, 'r') as f:
            db_list = json.load(f)
    else:
        db_list = []
    db_append_list = sorted(smiles_set - set(db_list))

    # logging.info(f"Append {len(smiles_list_with_none)}>{len(smiles_list)}>{len(smiles_set)}>{len(db_append_list)} to {len(db_list)}")
    db_list = db_list + db_append_list
    logging.info(f"Now {len(db_list)}")

    assert len(db_list) == len(set(db_list))

    with open(db_fn, 'w') as f:
        json.dump(db_list, f, indent=4)
    logging.info(f"Saved! smiles_db: {db_fn}")

    logging.info("Finish!")
    