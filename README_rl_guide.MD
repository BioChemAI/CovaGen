# CovaGen-guide, CovaGen-rl and CovaGen
This part is used for the training and sampling of CovaGen-rl and CovaGen-guide

# Usage
## Sampling
Use the script scripts/rl_train_sample_guided.py and scripts/rl_sample_guided_singleseq.py for sampling.
### Single protein sampling
For CovaGen-guide, use the model before RL fine-tuning, and specify the classifier scale s, and the classifier model path. 2 Classifiers are provided in ./Models/.
```
python scripts/rl_sample_guided_singleseq.py   --classifier_scale scale/s --protein_seq "sequence" --classifier_path path/to/classifier --model_path path/to/trained/model \\
    --save_path path/for/saving/SMILES/generated/pkl
    --class_cond False --rescale_timesteps False --diffusion_steps 300 --noise_schedule linear \\\
```

For CovaGen-rl, use the fine-tuned model, set classifier_scale to 0.
```
python scripts/rl_sample_guided_singleseq.py   --classifier_scale 0 --protein_seq "sequence" --classifier_path path/to/classifier --model_path path/to/rl/model
    --save_path path/for/saving/SMILES/generated/pkl
    --class_cond False --rescale_timesteps False --diffusion_steps 300 --noise_schedule linear
```
And to sample with CovaGen, use the fine-tuned model and specify the classifier scale s.
Note that you need to stop these scripts manually to finish the sampling procedure, which saves generated molecules while running.

### Sample for proteins in the CrossDocked2020 testset
Use scripts/rl_train_sample_guided.py instead, with same parameters except don't specify the protein sequence.
The processed Crossdocked2020 test set should be downloaded and moved to the Models folder here.

Note that you need to stop these scripts manually to finish the sampling procedure, where generated molecules are saved while running.

## Training
To fine-tune the model on your scoring function, please go to the training folder. Run this to start training, trained models are saved in the log dir.
```
python rl_train.py   --classifier_path path/to/classifier --model_path path/to/diffusion/model --vae_dir path/of/trained/vae 
    --log_dir path/for/saving/finetuned/pkl --data_dir path/of/crossdocked/dataset
    --class_cond False --rescale_timesteps False --diffusion_steps 300 --noise_schedule linear --classifier_scale 0 
```
