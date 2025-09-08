#  CovaGen-uncond & CovaGen-cond Usage Guide
# Sampling
## CovaGen-uncond
**Step 1**: Generate latent vectors (default 5000 vectors)
```
python scripts/pl_sample_t_uncond  --model_path ./Models/uncond_model004200.pt --save_path wherever/you/like.pkl --diffusion_steps 200 --noise_schedule linear --rescale_timesteps False
```
**Step 2**: Decode latent vectors to molecules
```
python scripts/decode_save_single.py --sampled_vec saved/path/ --save_path_10k path/to/valid/SMILES --save_path_full path/to/all/SMILES --vae_path ./Models/080_NOCHANGE_evenhigherkl.ckpt
```

## CovaGen-cond

### Single Protein Sampling
Input the protein sequence instead.
```
python scripts/pl_sample_singleseq.py  --model_path ../Models/model009000.pt  --save_path path/to/a/pklfile --num_samples amount/to/sample\\
       --protein_seq "Sequenc you want to sample with"  --diffusion_steps 300 --noise_schedule linear --rescale_timesteps False \\
```
Decode latent vectors into molecules
```
python scripts/decode_save_single.py --sampled_vec saved/path/ --save_path_10k path/to/valid/SMILES --save_path_full path/to/all/SMILES --vae_path ./Models/080_NOCHANGE_evenhigherkl.ckpt
```

### Batch Sampling (Crossdocked2020 Testset)
This will sample 200 latent vectors for each pocket in the testset, the processed Crossdocked2020 test set is provided in Models Folder.
```
python scripts/pl_sample_full.py --model_path ./Models/cond_model009000.pt --save_path path/to/a/directory! --diffusion_steps 300 --noise_schedule linear --rescale_timesteps False 
```
Decode them.
```
python scripts/decode_save.py --sampled_vec path/of/latent vectors --save_path_10k path/for/100valid/molecules/for/each/pocket --save_path_full path/for/all/decoded/molecules --vae_path ./Models/080_NOCHANGE_evenhigherkl.ckpt
```

# Evaluation
>Scripts for evaluation are provided in the repository under 'CovaGen/results'

For the evaluation of molecule generation metrics, we followed the exact same pipeline as proposed in MOSES.

For the calculation of Vina score,QED,SA,etc, we provide the code that calculates these metrics for the molecules generated for all the 100 pockets in the CrossDocked test set.
Make sure to download QVina2, ADFRsuite and install Openbable.
First, arrange the generated molecules by running the following three scripts sequentially :
```
python results/Docking/evaluate_arrange.py --decoded path to molecules saved by decode_save.py \\
                                    --save_path output path, in .json format                                  
```
```
python results/Docking/evaluate.collect.py  --sample-smiles-json  |output of evalate_arrange.py \\ 
                                    --collect-pkl   |output path, in .pickle format \\
                                    --data-dir  |path to your CrossDock2020 dataset folder \\ 
                                    --receptor protein  | \\
                                    --receptor-box pocket | \\
                                    --ref-from-smiles  | \\
                                    --index-protein ./results/Docking/protein_filepath.json
```
```
python results/Docking/evaluate.collect.smiles_db.py  --collect-pkl  |output of evaluate.collect.py \\
                                      --db-json |output path, in .json format
```
Then, run following scripts for calculation:
```
python results/Docking/evaluate.docking_score.py    --collect-pkl | \\
                                    --output-json /workspace/codes/Docking/docking_score/docking_tdiff_hisgen.json \\ 
                                    --docked-dir /workspace/codes/Docking/docked/tdiff_hisgen/
```
```
python results/Docking/evaluate.qed_sa.py   --db-json  |output of evaluate.collect.smiles_db.py
                            --output-json  |output path, in json format
```
This one is for diversity.
```
python results/Docking/evaluate.novelty_diversity.py   --collect-pkl  |output of evaluate.collect.py \\
                                       --index-json  |file index_sbdd.json is provided in folder \\
                                       --frag-ligand-pkl |file frag_ligand_sbdd.pkl is provided in folder \\
                                       --output-json  |output path, in .json format 
```
After these, run this for all results.
```
python results/Docking/evaluate.analysis.noretro.py --qed-sa-json  |output of evaluate.qed_sa.py \\
                                    --novelty-diversity-json  |output of evaluate.novelty_diversity.py \\
                                    --docking-score-json  |output of evaluate.docking_score.py \\
                                    --collect-pkl  |output of evaluate.collect.py    \\
                                    --output-json  |result output \\
                                    --hash-csv-out  |output of all molecules and thei   r corresponding metrics \\
                                    --ref-compare-to smiles  \\                   
```
