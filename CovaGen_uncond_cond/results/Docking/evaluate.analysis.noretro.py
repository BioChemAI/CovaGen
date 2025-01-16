"""
Analysis (Overall)

## Usage

```bash
model_type=cvaeProteinToLigand
train_id=20221115_105154_1b96_sbdd
postfix=_u100

debug="-m debugpy --listen 5678 --wait-for-client"

python $debug evaluate.analysis.overall.py \
    --ref-compare-to sdf \
    --qed-sa-json saved/common/qed_sa.json \
    --retro-star-json saved/common/retro_star.json \
    --novelty-diversity-json saved/$model_type/$train_id/evaluate/novelty_diversity$postfix.json \
    --docking-score-json saved/$model_type/$train_id/evaluate/docking_score$postfix.json \
    --collect-pkl saved/$model_type/$train_id/evaluate/collect$postfix.pkl \
    --output-json saved/$model_type/$train_id/evaluate/overall$postfix.json \
    --hash-csv-out saved/$model_type/$train_id/evaluate/overall.hash$postfix.csv

python $debug evaluate.analysis.overall.py \
    --ref-compare-to sdf \
    --qed-sa-json saved/common/qed_sa.json \
    --retro-star-json saved/common/retro_star.json \
    --novelty-diversity-json saved/$model_type/$train_id/evaluate/novelty_diversity$postfix.json \
    --docking-score-json saved/$model_type/$train_id/evaluate/docking_score$postfix.json \
    --collect-pkl saved/$model_type/$train_id/evaluate/collect$postfix.pkl \
    --output-json saved/$model_type/$train_id/evaluate/overall${postfix}_with_normal_ref.json \
    --hash-csv-out saved/$model_type/$train_id/evaluate/overall.hash${postfix}_with_normal_ref.csv \
    --hash-ref-csv-in saved/baseline/ligan_sbdd_protein/evaluate/overall.hash.csv
```
"""

import argparse
import logging
import pickle
import json
import pandas as pd
import numpy as np
from utils.common import init_logging, use_path
from evaluateUtils.EvaluateSrcHashSet import EvaluateSrcHashSet


def default_dump(obj):
	"""Convert numpy classes to JSON serializable objects."""
	if isinstance(obj, (np.integer, np.floating, np.bool_)):
		return obj.item()
	elif isinstance(obj, np.ndarray):
		return obj.tolist()
	else:
		return obj


def parse_hash_set(evaluate_src_hash_set: EvaluateSrcHashSet,
				   qed_sa_dict, docking_score_dict) -> pd.DataFrame:  # removed retro* pos arg

	hash_csv_dict = dict((k, []) for k in (
		"hash", "code", "type", "idx", "smiles",
		"qed", "sa", "lipinski", "logp",
		"docking_score",
		*(f"lipinski_rule_{i}" for i in range(5))
	))
	for evaluate_src_hash_item in evaluate_src_hash_set:
		for _from in evaluate_src_hash_item["_from_list"]:
			_hash = evaluate_src_hash_item["_hash"]
			_smiles = evaluate_src_hash_item["ligand_smiles"]

			hash_csv_dict["hash"].append(_hash)
			hash_csv_dict["code"].append(_from[0])
			hash_csv_dict["type"].append(_from[1])
			hash_csv_dict["idx"].append(_from[2])
			hash_csv_dict["smiles"].append(_smiles)

			hash_csv_dict["qed"].append(qed_sa_dict[_smiles]["qed"] if _smiles in qed_sa_dict.keys() else None)
			hash_csv_dict["sa"].append(qed_sa_dict[_smiles]["sa"] if _smiles in qed_sa_dict.keys() else None)
			hash_csv_dict["lipinski"].append(
				qed_sa_dict[_smiles]["lipinski"] if _smiles in qed_sa_dict.keys() else None)
			hash_csv_dict["logp"].append(qed_sa_dict[_smiles]["logp"] if _smiles in qed_sa_dict.keys() else None)


			hash_csv_dict["docking_score"].append(docking_score_dict[_hash]["docking_score"])

			for i in range(5):
				hash_csv_dict[f"lipinski_rule_{i}"].append(
					((1 if qed_sa_dict[_smiles]["lipinski_rules"][i] else 0) \
						 if qed_sa_dict[_smiles]["lipinski_rules"] else None) \
						if _smiles in qed_sa_dict.keys() else None
				)

	df = pd.DataFrame(data=hash_csv_dict)
	return df


def describe_median(df, method=["median"]):
	des = df.describe()
	des = des.append(df.reindex(df.columns, axis=1).agg(method))
	return des


def overall_statistics(df, novelty_diversity_dict):
	type_dict = dict()
	for _type in ("ref_sdf", "ref_smiles", "prd_smiles"):
		_dict = dict()
		sub_df = df.loc[df["type"] == _type]

		des = describe_median(
			sub_df.loc[:, ["hash", "code"]],
			["nunique", "count"]
		)
		_dict = dict(**_dict, **des.to_dict())

		des = describe_median(
			sub_df.groupby(by="code").mean().iloc[:, 1:],
			["median"]
		)
		_dict = dict(**_dict, **des.to_dict())

		type_dict[_type] = _dict

	high_dict = dict()
	# df.drop('retro_star',axis = 1)
	for _type in ("ref_sdf", "ref_smiles"):
		metric_dict_list = dict((k, []) for k in df.columns[5:])
		for code, group in df.groupby(by="code"):
			for metric, _list in metric_dict_list.items():
				try:
					sc_ref = group[group["type"] == _type][metric].item()
				except:
					sc_ref = None
				sc_prd_ds = group[group["type"] == "prd_smiles"][metric].dropna()
				_list.append((sc_prd_ds < sc_ref).sum() / sc_prd_ds.count())
		des = describe_median(
			pd.DataFrame(metric_dict_list),
			["median"]
		)
		high_dict[_type] = des.to_dict()

	dnty_dict_list = dict(diversity=[], scaffold_diversity=[])
	for code, group in df.groupby(by="code"):
		x = novelty_diversity_dict[code]

		for item in ("similarity_prd_comb", "similarity_ref_prd", "novelty_train_prd", "novelty_train_prd_smiles"):
			if item in dnty_dict_list.keys():
				dnty_dict_list[item].append(x[item])
			else:
				dnty_dict_list[item] = [x[item]]
		dnty_dict_list["diversity"].append(1 - x["similarity_prd_comb"])

		for item in ("similarity_prd_comb", "similarity_ref_prd", "novelty_train_prd"):
			scaffold_item = f"scaffold_{item}"
			if scaffold_item in dnty_dict_list.keys():
				dnty_dict_list[scaffold_item].append(x["scaffold"][item])
			else:
				dnty_dict_list[scaffold_item] = [x["scaffold"][item]]
		dnty_dict_list["scaffold_diversity"].append(1 - x["scaffold"]["similarity_prd_comb"])

	des = describe_median(
		pd.DataFrame(dnty_dict_list),
		["median"]
	)
	dnty_dict = des.to_dict()

	return dict(type=type_dict, high=high_dict, dnty=dnty_dict)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Analysis.')
	parser.add_argument("--ref-compare-to", choices=["smiles", "sdf"], default="sdf")
	parser.add_argument("--qed-sa-json", type=str)
	parser.add_argument("--novelty-diversity-json", type=str)
	parser.add_argument("--docking-score-json", type=str)
	parser.add_argument("--collect-pkl", type=str)
	parser.add_argument("--output-json", type=str)
	parser.add_argument("--hash-csv-in", type=str, default=None)
	parser.add_argument("--hash-csv-out", type=str, default=None)
	parser.add_argument("--hash-ref-csv-in", type=str, default=None)
	args = parser.parse_args()

	output_fn = use_path(file_path=args.output_json)
	pure_fn = use_path(file_path=f"{args.output_json}.csv")
	log_fn = use_path(file_path=f"{args.output_json}.log")

	init_logging(log_fn)
	logging.info(args)

	if args.hash_csv_in:

		df = pd.read_csv(args.hash_csv_in)
		logging.info(f"Loaded! hash_csv_in: {args.hash_csv_in}")

	else:
		collect_fn = use_path(file_path=args.collect_pkl, new=False)
		qed_sa_fn = use_path(file_path=args.qed_sa_json, new=False)
		novelty_diversity_fn = use_path(file_path=args.novelty_diversity_json, new=False)
		docking_score_fn = use_path(file_path=args.docking_score_json, new=False)

		with open(collect_fn, 'rb') as f:
			evaluate_src_hash_set: EvaluateSrcHashSet = pickle.load(f)
		logging.info(f"Loaded! evaluate_src_hash_set: {collect_fn}")

		# Load the following
		sample_smiles_list, ref_smiles_list = evaluate_src_hash_set._data

		with open(qed_sa_fn, 'r') as f:
			qed_sa_dict = json.load(f)
		logging.info(f"Loaded! qed_sa_dict: {qed_sa_fn}")


		with open(novelty_diversity_fn, 'r') as f:
			novelty_diversity_dict = json.load(f)
		logging.info(f"Loaded! novelty_diversity_dict: {novelty_diversity_fn}")

		with open(docking_score_fn, 'r') as f:
			docking_score_dict = json.load(f)
		logging.info(f"Loaded! docking_score_dict: {docking_score_fn}")

		# Hash CSV
		df = parse_hash_set(
			evaluate_src_hash_set,
			qed_sa_dict, docking_score_dict  # removed retro* dict
		)

	if args.hash_ref_csv_in:
		df_ref = pd.read_csv(args.hash_ref_csv_in)
		df = pd.concat([
			df[df["type"].isin(["prd_smiles"])],
			df_ref[df_ref["type"].isin(["ref_smiles", "ref_sdf"])]
		])
		logging.info(f"Loaded! hash_ref_csv_in: {args.hash_ref_csv_in}")

	logging.info("Analysis (Overall)...")

	if args.hash_csv_out:
		df.to_csv(args.hash_csv_out, index=False)
		logging.info(f"Saved! hash_csv_out: {args.hash_csv_out}")

	overall_dict = dict()
	overall_dict = overall_statistics(df, novelty_diversity_dict)

	logging.info(json.dumps(overall_dict, indent=4, default=default_dump))

	tab_list = []
	tab_list.append(''.join([
		(
			f"type\t"
			f"docking_score\t"
			f"qed\t"
			f"sa\t"
			f"lipinski\t"
			f"logp\t"
			f"lipinski_rules\t"
		),
		(
			f"h_docking_score\t"
			f"diversity\t"
			f"similarity_rp\t"
			f"novelty_tp\t"
			f"s_diversity\t"
			f"s_similarity_rp\t"
			f"s_novelty_tp\t"
			f"novelty_tp_smi"
		)
	]))


	def get_mean_std(path: str, per=False):
		obj = overall_dict
		for item in path.split('.'):
			obj = obj[item]
		if per:
			return f"{obj['mean'] * 100:.2f} ± {obj['std'] * 100:.2f}"
		else:
			return f"{obj['mean']:.3f} ± {obj['std']:.3f}"


	ref_type = f"ref_{args.ref_compare_to}"
	for _idx, _type in enumerate((ref_type, "prd_smiles")):
		lipinski_rules_str = ' '.join(
			f"{overall_dict['type'][_type][f'lipinski_rule_{j}']['mean'] * 100:.2f}" \
			for j in range(5))
		tab_list.append(
			''.join((
				'\t'.join((
					f"{_type}",
					get_mean_std(f'type.{_type}.docking_score'),
					get_mean_std(f'type.{_type}.qed'),
					get_mean_std(f'type.{_type}.sa'),
					get_mean_std(f'type.{_type}.lipinski'),
					get_mean_std(f'type.{_type}.logp'),
					f"{lipinski_rules_str}"
				)),
				'\t'.join((
					f"",
					get_mean_std(f'high.{ref_type}.docking_score', per=True),

					get_mean_std(f'dnty.diversity'),
					get_mean_std(f'dnty.similarity_ref_prd'),
					get_mean_std(f'dnty.novelty_train_prd', per=True),

					get_mean_std(f'dnty.scaffold_diversity', per=True),
					get_mean_std(f'dnty.scaffold_similarity_ref_prd'),
					get_mean_std(f'dnty.scaffold_novelty_train_prd', per=True),

					get_mean_std(f'dnty.novelty_train_prd_smiles', per=True)
				)) if _idx > 0 else f"\t" * 8
			))
		)

	logging.info('tab:\n' + '\n'.join(tab_list))

	with open(pure_fn, 'w') as f:
		f.write('\n'.join(x.replace('\t', ',') for x in tab_list))
	logging.info(f"Saved! pure_csv: {pure_fn}")

	logging.info(
		f"valid: {int(overall_dict['type']['prd_smiles']['qed']['count'])}, {int(overall_dict['type']['prd_smiles']['sa']['count'])}")

	with open(output_fn, 'w') as f:
		json.dump(overall_dict, f, indent=4, default=default_dump)
	logging.info(f"Saved! output_json: {output_fn}")

	logging.info("Finish!")
