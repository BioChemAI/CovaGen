"""
Parallel docking

## Usage

```bash
model_type=egnnCvaeProteinToLigand
train_id=20221102_170634_f481_sbdd

python evaluate.docking_score.py \
    --collect-pkl saved/$model_type/$train_id/evaluate/collect.pkl \
    --output-json saved/$model_type/$train_id/evaluate/docking_score.json \
    --docked-dir saved/$model_type/$train_id/evaluate/docked \
    --chunk-size 256 \
    --n-jobs 32 \
    --qvina-cpu 8
```
"""

import argparse
import logging
from pathlib import Path    #
import time
import pickle
import json
from typing import Callable, Dict
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.common import chunks, init_logging, mean_without_none, statistics_to_str, use_path
from evaluateUtils.docking_score import docking_score
from evaluateUtils.EvaluateSrcHashSet import EvaluateSrcHashSet

def docking_par(evaluate_src_hash_set: EvaluateSrcHashSet, result_dict: Dict, docked_dn: Path,
    chunk_size: int, qvina_cpu: int, n_jobs: int, save_fun: Callable):

    def par_g(evaluate_src_chunk):
        for evaluate_src in evaluate_src_chunk:
            yield (evaluate_src["_hash"], evaluate_src["receptor_pdb"], evaluate_src["ligand_mol"], 
            evaluate_src.get("box", None), evaluate_src.get("more_kwargs", dict()))

    def par_f(_hash, protein_pdb, ligand_mol, box, more_kwargs):
        tic = time.time()
        try:
            _docking_score, _docked_pdbqt = docking_score(protein_pdb, ligand_mol, box=box, cpu=qvina_cpu, _hash=_hash, **more_kwargs)
        except:
            _docking_score, _docked_pdbqt = None, None
        _docking_score_toc = time.time() - tic
        return _docking_score, _docked_pdbqt, _docking_score_toc

    evaluate_src_chunk_list = list(chunks(evaluate_src_hash_set, chunk_size))
    evaluate_src_chunk_pbar = tqdm(evaluate_src_chunk_list, desc="Evaluate [docking_score]")

    for evaluate_src_chunk in evaluate_src_chunk_pbar:
        # Bypass the processed part
        evaluate_src_chunk = list(filter(
            lambda x: not (result_dict.get(x["_hash"], None) and result_dict[x["_hash"]]["docking_score"]), 
            evaluate_src_chunk))
        if len(evaluate_src_chunk) > 0:
            result_chunk = Parallel(n_jobs=n_jobs)(delayed(par_f)(*x) for x in par_g(evaluate_src_chunk))
            _docking_score_list, _docked_pdbqt_list, _docking_score_toc_list = list(zip(*result_chunk))

            for evaluate_src, _docking_score, _docked_pdbqt, _docking_score_toc in \
                zip(evaluate_src_chunk, _docking_score_list, _docked_pdbqt_list, _docking_score_toc_list):

                _hash = evaluate_src["_hash"]
                result_dict[_hash] = dict(docking_score=_docking_score, docking_score_toc=_docking_score_toc)

                if (_docked_pdbqt is not None):
                    with open(docked_dn / f"{_hash}.pdbqt", 'wb') as f:
                        f.write(_docked_pdbqt)

            evaluate_src_chunk_pbar.set_postfix({
                "len_chunk": len(evaluate_src_chunk),
                "success_chunk": len([x for x in _docking_score_list if x is not None]), 
                "docking_score": mean_without_none(_docking_score_list), 
                "docking_score_toc": mean_without_none(_docking_score_toc_list)
            })

        save_fun(result_dict)
    save_fun(result_dict)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Docking.')
    parser.add_argument("--collect-pkl", type=str)
    parser.add_argument("--output-json", type=str)
    parser.add_argument("--docked-dir", type=str)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--qvina-cpu", type=int, default=8)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--hash-list", nargs='+', type=str, default=[])
    args = parser.parse_args()

    #读入前一步处理的
    collect_fn = use_path(file_path=args.collect_pkl, new=False)

    #三个输出的路径
    output_fn = use_path(file_path=args.output_json)
    docked_dn = use_path(dir_path=args.docked_dir)
    log_fn = use_path(file_path=f"{args.output_json}.log")
    
    init_logging(log_fn)
    logging.info(args)

    #读入数据
    with open(collect_fn, 'rb') as f:
        evaluate_src_hash_set: EvaluateSrcHashSet = pickle.load(f)
    logging.info(f"Loaded! evaluate_src_hash_set: {collect_fn}")
    ls = []
    for i in evaluate_src_hash_set._list:
        ls.append(i['ligand_smiles'])
    with open('/workspace/mpro_gen.pkl','wb')as f:
        pickle.dump(ls,f)
    #继续还是新建开始的字典
    if output_fn.exists():
        with open(output_fn, 'r') as f:
            result_dict = json.load(f)
        logging.info(f"Continue job! result_dict: {output_fn}")
    else:
        result_dict = dict()
        logging.info(f"New job!")

    logging.info("Estimate Docking Score...")

    #这样存出来result_dict
    def save_fun(result_dict):
        with open(output_fn, 'w') as f:
            json.dump(result_dict, f)

    if len(args.hash_list) > 0:
        evaluate_src_hash_set._list = [x for x in evaluate_src_hash_set._list if x["_hash"] in args.hash_list]

    docking_par(
        evaluate_src_hash_set if not args.dev else evaluate_src_hash_set[:5], 
        result_dict,
        docked_dn,
        chunk_size = args.chunk_size,
        qvina_cpu = args.qvina_cpu,
        n_jobs = args.n_jobs,
        save_fun = save_fun
    )

    logging.info(f"Saved! result_dict: {output_fn}")

    docking_score_list = [x["docking_score"] for _, x in result_dict.items() if x["docking_score"] is not None]
    logging.info(f"Done! {statistics_to_str(docking_score_list, 'docking_score')}")

    def statistics_by_type(_type):
        _hash_list = []
        for x in evaluate_src_hash_set:
            _, _types, _ = zip(*x["_from_list"])
            if _type in _types:
                _hash_list.append(x["_hash"])
        _score_list = [x["docking_score"] for _hash, x in result_dict.items() if x["docking_score"] is not None and _hash in _hash_list]
        logging.info(f"Done! {statistics_to_str(_score_list, f'docking_score_{_type}')}")

    for _type in ("ref_sdf", "ref_smiles", "prd_smiles"):
        statistics_by_type(_type)

    logging.info("Finish!")
    