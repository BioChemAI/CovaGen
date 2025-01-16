"""
Sampling latent vectors and save. This is based on classifier_sample.py in the implementation of guided-diffusion
"""

import argparse
import pickle
import os, sys

sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from datetime import datetime, date
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
	NUM_CLASSES,
	model_and_diffusion_defaults,
	classifier_defaults,
	create_model_and_diffusion,
	create_classifier,
	add_dict_to_argparser,
	args_to_dict,
)
from guided_diffusion.qed_models import Net, Net2, condtimeNet, reg_Net,reg_Net_newt,deeper_cls_net
from optdiffusion.model import Dynamics_samp, Dynamics_t_samp, Dynamics_t_samp2, Dynamics_t,simple_reg_Net_newt,Dynamics_t_esm_deeper
from guided_diffusion.pl_datasets import load_data_smi,load_data_esm
from guided_diffusion.train_util import TrainLoopRL



def main():
	args = create_argparser().parse_args()

	dist_util.setup_dist()
	logger.configure(dir='./logs/')

	logger.log("creating model and diffusion...")
	_, diffusion = create_model_and_diffusion(
		**args_to_dict(args, model_and_diffusion_defaults().keys())
	)
	_, prior_diffusion = create_model_and_diffusion(
		**args_to_dict(args, model_and_diffusion_defaults().keys())
	)



	agent_model = Dynamics_t_esm_deeper(condition_dim=28, target_dim=128, hid_dim=64, condition_layer=3, n_heads=2,
							 condition_time=True, sampling=True)
	prior_model = Dynamics_t_esm_deeper(condition_dim=28, target_dim=128, hid_dim=64, condition_layer=3, n_heads=2,
							 condition_time=True, sampling=True)

	agent_model.load_state_dict(
		dist_util.load_state_dict(args.model_path, map_location="cpu")
	)
	prior_model.load_state_dict(
		dist_util.load_state_dict(args.model_path, map_location="cpu")
	)

	agent_model.to(dist_util.dev())
	prior_model.to(dist_util.dev())
	agent_model.train()
	prior_model.eval()
	# Load scoring model
	classifier = deeper_cls_net()
	classifier.load_state_dict(
		dist_util.load_state_dict(args.classifier_path, map_location="cpu")
	)
	classifier.to(dist_util.dev())
	classifier.eval()

	def cond_fn(x, t,y=None):
		assert y is not None
		with th.enable_grad():
			x_in = x.detach().requires_grad_(True)
			logits = classifier(x_in, t)
			log_probs = F.log_softmax(logits, dim=-1)
			# y = y.repeat(3000)
			selected = log_probs[range(len(logits)), y.view(-1)]
			return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale


	dat = load_data_esm(  #
		data_dir=args.data_dir,
		batch_size=args.batch_size,
		vae_dir=args.vae_dir,
		dataset_save_path=args.dataset_save_path,
		data_state="train",
        sequence=args.protein_seq
	)

	TLLOOP = TrainLoopRL(
		model=agent_model,
		prior=prior_model,
		diffusion=diffusion,
		data=dat,
		batch_size=args.batch_size,
		lr=args.lr,  # adjust
		ema_rate=args.ema_rate,
		log_interval=args.log_interval,
		save_interval=args.save_interval,
		resume_checkpoint=args.resume_checkpoint,
		use_fp16=args.use_fp16,
		fp16_scale_growth=args.fp16_scale_growth,
		weight_decay=args.weight_decay,
		lr_anneal_steps=args.lr_anneal_steps,
		scoring_model=classifier
	)

	diffusion.insert_rlloop(TLLOOP)

	logger.log("Starting an RL Train...")
	time_start = datetime.now()
	cnt=0
	for batch in dat:

		model_kwargs = {}
		classes = th.tensor(np.ones((100,), dtype=np.int)).to('cuda:0')
		model_kwargs["y"] = classes
		sample_fn = (
			diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
		)

		sample = sample_fn(
			agent_model,
			(100, 128),
			batch=batch,
			cond_fn=cond_fn,
			clip_denoised=args.clip_denoised,
			model_kwargs=model_kwargs,
			device=dist_util.dev(),
			progress=True,
			saving_path = args.save_path,

		)

		cnt+=1

		time_end = datetime.now()
		dist.barrier()
	time_samp = (time_end - time_start).seconds


def create_argparser():
	defaults = dict(
		clip_denoised=False,
		num_samples=2000,
		batch_size=100,
		use_ddim=False,
		data_dir = '/data',
		vae_dir = '../Models/080_NOCHANGE_evenhigherkl.ckpt',
		model_path="",
		classifier_path="",
		classifier_scale=0,
		save_path="",#
		data_path="/data/",
		dataset_save_path="/data/",
		processed_filename='',
		processed_name2id_name='',
		lr =7e-4,#
		ema_rate="0.9999",
		log_interval=10,#
		save_interval=1000,
		resume_checkpoint="",
		use_fp16=False,
		fp16_scale_growth=1e-3,
		weight_decay=0.0,
		lr_anneal_steps=0,
        protein_seq = "",
	)
	defaults.update(model_and_diffusion_defaults())
	defaults.update(classifier_defaults())
	parser = argparse.ArgumentParser()
	add_dict_to_argparser(parser, defaults)
	return parser


if __name__ == "__main__":
	main()
