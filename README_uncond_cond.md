# CovaGen-uncond and CovaGen-cond

# Usage

This section is about sampling and training procedure of CovaGen-uncond and CovaGen-cond

# CovaGen-uncond
Unconditional generation, first sample and save latent vectors,this generates 5000 latent vectors
```
python scripts/pl_sample_t_uncond  --model_path ./Models/uncond_model004200.pt --save_path wherever/you/like.pkl --diffusion_steps 200 --noise_schedule linear --rescale_timesteps False
```
Decode latent vectors into molecules
```
python scripts/decode_save_single.py --sampled_vec saved/path/ --save_path_10k path/to/valid/SMILES --save_path_full path/to/all/SMILES --vae_path ./Models/080_NOCHANGE_evenhigherkl.ckpt
```
CovaGen
# CovaGen-cond

## Sample for a single protein target
Input the protein sequence instead.
```
python scripts/pl_sample_singleseq.py  --model_path ../Models/model009000.pt  --save_path path/to/a/pklfile --num_samples amount/to/sample\\
       --protein_seq "Sequenc you want to sample with"  --diffusion_steps 300 --noise_schedule linear --rescale_timesteps False \\
```
Decode latent vectors into molecules
```
python scripts/decode_save_single.py --sampled_vec saved/path/ --save_path_10k path/to/valid/SMILES --save_path_full path/to/all/SMILES --vae_path ./Models/080_NOCHANGE_evenhigherkl.ckpt
```

## Sample for all pockets in the crossdocked2020 testset
This will sample 200 latent vectors for each pocket in the testset, the processed Crossdocked2020 test set is provided in Models Folder.
```
python scripts/pl_sample_full.py --model_path ./Models/cond_model009000.pt --save_path path/to/a/directory! --diffusion_steps 300 --noise_schedule linear --rescale_timesteps False 
```
Decode them.
```
python scripts/decode_save.py --sampled_vec path/of/latent vectors --save_path_10k path/for/100valid/molecules/for/each/pocket --save_path_full path/for/all/decoded/molecules --vae_path ./Models/080_NOCHANGE_evenhigherkl.ckpt
```

### Evaluation
>Scripts for evaluation are provided in the repository under 'CovaGen/results'

For the evaluation of molecule generation metrics, we followed the exact same pipeline as proposed in MOSES.

For the calculation of Vina score,QED,SA,etc, we provide the code that calculates these metrics for the molecules generated for all the 100 pockets in the CrossDocked test set.
Make sure to download QVina2, ADFRsuite and install Openbable.
First, arrange the generated molecules by running the following three scripts sequentially :
```
python results/evaluate_arrange.py --decoded path to molecules saved by decode_save.py \\
                                    --save_path output path, in .json format                                  
```
```
python results/evaluate.collect.py  --sample-smiles-json  |output of evalate_arrange.py \\ 
                                    --collect-pkl   |output path, in .pickle format \\
                                    --data-dir  |path to your CrossDock2020 dataset folder \\ 
                                    --index-json  |The file index.json is provided in folder.  
                                    --receptor protein  | \\
                                    --receptor-box pocket | \\
                                    --ref-from-smiles  | \\
                                    --index-protein the file protein_filepath.json is provided in folder.
```
```
python evaluate.collect.smiles_db.py  --collect-pkl  |output of evaluate.collect.py \\
                                      --db-json |output path, in .json format
```
Then, run following scripts for calculation:
```
python evaluate.docking_score.py    --collect-pkl | \\
                                    --output-json /workspace/codes/Docking/docking_score/docking_tdiff_hisgen.json \\ 
                                    --docked-dir /workspace/codes/Docking/docked/tdiff_hisgen/
```
```
python evaluate.qed_sa.py   --db-json  |output of evaluate.collect.smiles_db.py
                            --output-json  |output path, in json format
```
This one is for diversity.
```
python evaluate.novelty_diversity.py   --collect-pkl  |output of evaluate.collect.py \\
                                       --index-json  |file index_sbdd.json is provided in folder \\
                                       --frag-ligand-pkl |file frag_ligand_sbdd.pkl is provided in folder \\
                                       --output-json  |output path, in .json format 
```
After these, run this for all results.
```
python evaluate.analysis.noretro.py --qed-sa-json  |output of evaluate.qed_sa.py \\
                                    --novelty-diversity-json  |output of evaluate.novelty_diversity.py \\
                                    --docking-score-json  |output of evaluate.docking_score.py \\
                                    --collect-pkl  |output of evaluate.collect.py    \\
                                    --output-json  |result output \\
                                    --hash-csv-out  |output of all molecules and thei   r corresponding metrics \\
                                    --ref-compare-to smiles  \\                   
```
### Training
To train unconditional latent diffusion model, go to the training folder, make sure you have all the dataset and models downloaded and put in the Models folder, then run
```
python pl_train_uncond_zinc_t.py    --logging_path model/saving/path --dataset_save_path uncond/dataset/path --vae_dir path/to/vae
                                       --diffusion_steps 200 --noise_schedule linear 
                                    --schedule_sampler loss-second-moment --batch_size 65536 --lr 1e-3 --rescale_timesteps False
```
And to train a conditional latent diffusion model
```
python pl_train_full.py         --log_dir model/save/path --vae_dir vae/path --data_dir crossdocked/dataset/path
                                --dataset_save_path processed/crossdocked/path
                                --diffusion_steps 300 --noise_schedule linear --schedule_sampler uniform --batch_size 128
```
