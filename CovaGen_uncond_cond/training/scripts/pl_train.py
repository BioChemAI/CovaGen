"""
Try to Train a diffusion model on Smiles.
"""

import argparse
import os,sys
sys.path.append(os.path.dirname(sys.path[0]))
import torch
from improved_diffusion import dist_util, logger
from improved_diffusion.pl_datasets import load_data_smi
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from optdiffusion import model as opmd
sys.path.append(os.path.dirname(sys.path[0]))

def main():
    args = create_argparser().parse_args()
     
    dist_util.setup_dist() 
    logger.configure(dir="/dataset/grid_search/cond_old_3000steps/") 
    logger.log("creating model and diffusion...")
    _1, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    ) 
    
    
    model = opmd.Dynamics(condition_dim=28, target_dim=128, hid_dim=64, condition_layer=3, n_heads=2, condition_time=True)
    model.to(dist_util.dev()) 
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    
    

    logger.log("creating data loader...")
    data = load_data_smi(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        vae_dir=args.vae_dir,
        dataset_save_path=args.dataset_save_path,
        data_mood="train"
    ) 

    logger.log("training...") 
    
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size, 
        microbatch=args.microbatch, 
        lr=args.lr, 
        ema_rate=args.ema_rate, 
        log_interval=args.log_interval, 
        save_interval=args.save_interval, 
        resume_checkpoint=args.resume_checkpoint, 
        use_fp16=args.use_fp16, 
        fp16_scale_growth=args.fp16_scale_growth, 
        schedule_sampler=schedule_sampler, 
        weight_decay=args.weight_decay, 
        lr_anneal_steps=args.lr_anneal_steps, 
    ).run_loop()



def create_argparser():
    '''自动地从字典中生成命令行传参的argumentparser，免去了手打的反复'''
    defaults = dict( 
        data_dir="/dataset/crossdock/crossdocked_pocket10",
        vae_dir="/workspace/codes/vaemodel/070_rnnattn-256_zinc.ckpt",
        dataset_save_path="/dataset/crossdock/processed",
        schedule_sampler="uniform", 
        lr=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=256,
        microbatch=-1,  
        ema_rate="0.9999",  
        log_interval=10,
        save_interval=1000,
        resume_checkpoint="", 
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults()) 
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    return parser 


if __name__ == "__main__":
    main()
