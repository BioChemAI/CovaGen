"""
Estimate QED & SA

## Usage

```bash
python evaluate.qed_sa.py \
    --db-json saved/common/smiles_db.json \
    --output-json saved/common/qed_sa.json
```
"""

import argparse
import logging
import time
import json
from typing import Dict, List
from tqdm import tqdm

from utils.common import init_logging, statistics_to_str, use_path
from evaluateUtils.qed_sa import QedSa

def qed_sa_sin(smiles_list: List[str]) -> Dict[str, Dict[str, float]]:
    """Output dict like smiles: {qed: float, sa: float}"""

    smiles_pbar = tqdm(smiles_list, desc="Evaluate [qed_sa]")
    result_dict = dict()
    for smiles_idx, smiles in enumerate(smiles_pbar):
        tic = time.time()
        qed_sa = QedSa(smiles=smiles)
        _qed, _sa, _logp = qed_sa.qed(), qed_sa.sa(), qed_sa.logp()
        _lipinski, _lipinski_rules = qed_sa.lipinski(return_rules=True)
        _toc = time.time() - tic
        result_dict[smiles] = dict(qed=_qed, sa=_sa, lipinski=_lipinski, logp=_logp, 
            toc=_toc, lipinski_rules=_lipinski_rules)
        if (smiles_idx + 1) % 100 == 0:
            smiles_pbar.set_postfix(result_dict[smiles])
    return result_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Estimate QED & SA.')
    parser.add_argument("--db-json", type=str)
    parser.add_argument("--output-json", type=str)
    args = parser.parse_args()

    db_fn = use_path(file_path=args.db_json)
    output_fn = use_path(file_path=args.output_json)
    log_fn = use_path(file_path=f"{args.output_json}.log")
    
    init_logging(log_fn)
    logging.info(args)

    logging.info(f"New job!")
    logging.info("Estimate QED & SA...")

    with open(db_fn, 'r') as f:
        db_list = json.load(f)

    result_dict = qed_sa_sin(db_list)

    with open(output_fn, 'w') as f:
        json.dump(result_dict, f, indent=4)
    logging.info(f"Saved! result_dict: {output_fn}")

    for item in ("qed", "sa", "lipinski", "logp"):
        _list = [x[item] for _, x in result_dict.items() if x[item] is not None]
        logging.info(f"Done! {statistics_to_str(_list, item)}")

    logging.info("Finish!")
    