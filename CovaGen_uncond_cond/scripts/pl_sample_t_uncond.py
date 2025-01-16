
import argparse
import os,sys
import time
import pickle
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
from improved_diffusion.pl_datasets import load_data_smi
from datetime import datetime, date


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    _1, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model = opmd.Dynamics_t_uncond_deeper(condition_dim=28,target_dim=128,hid_dim=64,condition_layer=3,n_heads=2,condition_time=True,sampling=True)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    data = None
    allsecs = 0
    for i in range(0, args.num_samples):
        sm_list = []
        time_1 = datetime.now()
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (5000, 128),
            batch = data,
            clip_denoised=args.clip_denoised,
            num_samples=args.num_samples,
            progress = True,
        )
        sm_list.append(sample)
        print("shape of a out:",sample.shape)
        print("out 1:",sample[0])
        print(sm_list)
        print(len(sm_list))
        time_2 = datetime.now()
        seconds = (time_2 - time_1).seconds
        print(f"sample {i+1} done,takes {seconds} seconds")
        allsecs += seconds
        break
    with open(args.save_path,"wb") as f1:
        pickle.dump(sample,f1)
    print(f"sampling done, all of the {args.num_samples} took {allsecs} secs")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=100,
        batch_size=100,
        use_ddim=False,
        data_dir="",
        dataset_save_path="",
        model_path="",
        save_path= ""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    sample_path = ""
    main()#
