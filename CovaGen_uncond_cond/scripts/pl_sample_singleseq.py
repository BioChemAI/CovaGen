"""
Try to generate some(maybe just one) ligand sample(s) with the trained model.
Here the way of sampling had been modified a bit: Sampling is always done by sampling the 100 test proteins at the same time.
    However, originally, the initial noise(the first one which is sample directly from gaussian.) is repeated x time to batch.
    for example 100 noise is sampled and it is repeated 100times so all of the noises make up for the final 10k output.
    But this version of code, all the noises is different vectors sampled from the gaussian.
"""
import argparse
import os,sys
import time
import pickle

import torch

sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import torch as th
import torch.distributed as dist


from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from optdiffusion import model as opmd
from improved_diffusion.pl_datasets import load_data_smi,load_data_esm
from datetime import datetime, date
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


set_seed(114514)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    _1, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model = opmd.Dynamics_t_esm_deeper(condition_dim=28,target_dim=128,hid_dim=64,condition_layer=3,n_heads=2,condition_time=True,sampling=True)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    data = load_data_esm(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        vae_dir=args.vae_dir,
        dataset_save_path=args.dataset_save_path,
        data_state="sample",
        sequence=args.protein_seq
    )
    allsecs = 0
    i=0
    sm_list = []
    os.makedirs(args.save_path, exist_ok=True)
    for dt in data:
        if len(sm_list) >= (args.num_samples // 200):
            break
        k = dt
        time_1 = datetime.now()
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (200, 128),
            batch = k,
            clip_denoised=args.clip_denoised,
            num_samples=args.num_samples,
            same_start=args.same_start,
        )
        sm_list.append(sample)
        print("shape of a out:",sample.shape)
        time_2 = datetime.now()
        seconds = (time_2 - time_1).seconds
        allsecs += seconds

    print(f"sampling done, all of the {args.num_samples} took {allsecs} secs")
    sm_list = torch.cat(sm_list, dim=0)
    with open(args.save_path + f"/sample_result.pkl", 'wb') as f:
        pickle.dump(sm_list, f)


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1000,
        batch_size=1,
        use_ddim=False,
        data_dir="/data/",
        dataset_save_path="/data/",
        model_path = "",
        save_path= "",
        protein_seq = "",
        vae_dir = "",
        same_start=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    seed=114514
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    main()#