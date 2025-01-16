
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
from improved_diffusion.pl_datasets import load_data_smi,load_data_esm
from datetime import datetime, date
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
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
    data = load_data_esm(  #
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        #vae_dir=args.vae_dir,
        dataset_save_path=args.dataset_save_path,
        data_state="sample"
    )
    sampled_dict = {}
    time_start = datetime.now()
    allsecs = 0
    i=0
    os.makedirs(args.save_path, exist_ok=True)
    for dt in data:
        k = dt
        if args.same_start:
            sample_amount = args.num_samples
        else:
            sample_amount = args.num_samples*50

        sm_list = []
        time_1 = datetime.now()
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sv = f"{args.save_path}" + f"{i+1}th"+ "_" + f"{args.diffusion_steps}steps" + ".pkl"
        sample = sample_fn(
            model,
            (args.num_samples, 128),
            batch = k,
            clip_denoised=args.clip_denoised,
            num_samples=args.num_samples,
            same_start=args.same_start,
        )
        sm_list.append(sample)
        print("shape of a out:",sample.shape)
        print("out 1:",sample[0])
        print(sm_list)
        print(len(sm_list))
        time_2 = datetime.now()
        seconds = (time_2 - time_1).seconds
        print(f"sample {i+1} done,takes {seconds} seconds")
        with open(args.save_path + f"/samp{i}th.pkl", 'wb') as f:
            pickle.dump(sample, f)
        i += 1
        if i == 100:
            break
        allsecs += seconds

    print(f"sampling done, all of the {args.num_samples} took {allsecs} secs")



def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=200,
        batch_size=1,
        use_ddim=False,
        data_dir="/data/",
        dataset_save_path="/data/",
        model_path = "",
        save_path= "",
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
    main()