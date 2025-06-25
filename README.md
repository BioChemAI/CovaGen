# CovaGen

This is the implementation of the paper "A Deep Generative Approach to \textit{de novo} Covalent Drug Design with Enhanced Drug-likeness and Safety".
There are two seperate folders, one implements CovaGen-uncond and CovaGen-cond, and the other one is CovaGen-guide and CovaGen-rl.

### This is the overall README, please refer to README_uncond_cond.md and README_rl_guide.md for a more detailed usage instruction

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Develop](#develop)
  - [Train](#train)
  - [Inference](#inference)
  - [Dataset](#dataset)
  - [Models and Files](#Models-and-Files)
## Installation

```bash
pip install -r requirements.txt
```

## Usage
For a more detailed usage instruction, please refer to the README_uncond_cond.md and README_rl_guide.md 

### Train
>The trained models are provided in the repository at 'CovaGen/CovaGen_uncond_cond/Models' and 'CovaGen/CovaGen_rl_guide/Models'.

### Inference
Please refer to the README_uncond_cond.md and README_rl_guide.md 

### Dataset
#### CrossDocked Dataset (Index by [3D-Generative-SBDD](https://github.com/luost26/3D-Generative-SBDD))

Download from the compressed package we provide <https://figshare.com/articles/dataset/crossdocked_pocket10_with_protein_tar_gz/25878871>.
```bash
tar xzf crossdocked_pocket10_with_protein.tar.gz
```
The following files are required to exist:
- `$sbdd_dir/split_by_name.pt`
- `$sbdd_dir/index.pkl`

### Models and Files

Trained models can be downloaded from here:
https://figshare.com/projects/DiffCDD/232655


After download, move the files in each folder to their corresponding Models folder.
