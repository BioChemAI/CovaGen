import copy
import functools
import os
import pickle
import time
# import jax#3
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0 # 或许loss step采样t会用到这个global的参数？注意后面用到了的话，这个是要调整的超参之一！？！！


class RL_TrainLoop:
    '''
    不对，它是在sampling的过程中train，那么，我应该是每次用train set中的多个pocket进行sampling。
    '''
    def __init__(
        self,
            *,  # 参数中星号是？
            model,
            prior,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            uncond=False,
                 ):
        self.prior = prior # 这是prior,创建的一个和agent相同的,不去更新参数的网络
        self.agent = model # 这是agent,是train好的一个diffusion
        self.diffusion = diffusion # diffusion框架
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.uncond = uncond

        # 上面这个空行的含义应该就是，前面是传入的参数组成的属性，且基本不会再更改。后面是自带的属性，以及是调用对象时会被改变的属性？
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()  # global batch？ get_world_size？
        print("model's parameters:", model.parameters())
        self.model_params = list(self.model.parameters())  # TODO
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()  # 这个参数用到哪儿了呢？sync_cuda
        self.ldict = []

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def RLloop(self):
        '''
        :return:
        '''
        # 在这里，接受每步得到的μ，σ和xt，用它们更新模型的参数，在200steps（暂定）更新完成后，进入下一个batch？
        # 也就是说，SampleLoop正常进行，在其后加这个RLLoop以更新参数
        #TODO 是在200step后更新prior，还是在每一步都去更新prior？
        



class TrainLoop: # 用类名来创建对象时，实际就是在调用构造函数__init__。
    def __init__(
        self,
        *, # 参数中星号是？
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        uncond = False,
    ):
        self.model = model # model 就是 create_model_and_diffusion创建出的一个Unet模型。
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        ) # 这是AdamW优化器的参数。如果只给ema_rate传入一个float值，那么它这里就是这一个float值的列表；否则传入一个“0.8,0.7”这种逗号分隔形式的字符串。
        # 这种对ema_rate的参数传入的方法应该是一种默认方法了吧，不然应该是会直接说明好的。
        # 注意这种短条件判断的用法。字符串的split方法把前面的字符串以指定的字符分隔。
        # 注意，这里把创建对象时传入的参数都传给了对象自己的属性，这样应该会方便对一些参数做另外的处理。
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.uncond = uncond

        # 上面这个空行的含义应该就是，前面是传入的参数组成的属性，且基本不会再更改。后面是自带的属性，以及是调用对象时会被改变的属性？
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size() # global batch？ get_world_size？
        print("model's parameters:", model.parameters())
        self.model_params = list(self.model.parameters()) # TODO
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available() # 这个参数用到哪儿了呢？sync_cuda
        self.ldict = []

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        print(f"use lr:{self.lr}")
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available(): # 分布式训练的初始化模块
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()], # 我把cudavisible写多几个是不是可以多卡train了？？我去原来原先代码里是自带多卡train的啊？？我是不是把这个给删掉了啊
                output_device=dist_util.dev(),  # 我相信output_device 就是相当于最后汇总数据的那块卡，难道不对这个dev()方法取list，他返回的就是第一个元素了吗？
                broadcast_buffers=False,    # 一定要去看看dist_util.dev()方法会返回怎么样的GPUlist，这多卡train不是美滋滋？
                bucket_cap_mb=128,
                find_unused_parameters=False, # 回去看看这ddp的整个方法的用法，争取以后能用上！？
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        ) # 回去一定好好看看blobfile的文档，他一直用blobfile的原因是？和原先的file的区别是？
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self): # 这是train的启动
        while ( # 如果不设置lr annealing步数这个while就是个while true， 为了一直循环训练。
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            timestr = time.strftime("%Y%m%d-%H%M%S")
            batch = next(self.data)
            #batch = batch.to(dist_util.dev())
            # if not self.uncond:
            #     if(batch.shape) != 4: #
            #         print("received a mal batch!!!")
            #         continue
            self.run_step(batch) # esm:batch[0]:padded token reps, batch[1]:masks,batch[2]:x.
            if self.step % self.log_interval == 0:
                logger.dumpkvs() # 10步一次dumpkvs，我不会在tmp里写了一堆logfiles吧？？去看看。
            if self.step % self.save_interval == 0:
                self.save() # save方法是iddpm中写的，现在是1000步存一次模型。
                # with open(f"/workspace/codes/middpm_uncond_zinc_t_loss_50_lrup/loss_{self.step}steps.pkl","wb") as f:
                #     pickle.dump(self.ldict,f)
                # Run for a finite amount of time in integration tests.
                # 下面这两句是为了在测试代码时运行有限次。environ
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                  return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond = None):#
        self.forward_backward(batch)
        # forwarbackward完成了lossbackward
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal() # 这里完成optimizer_step，完成一轮参数的更新#
        self.log_step()

    def forward_backward(self, batch, cond = None):
        zero_grad(self.model_params) # 梯度置零
        if not self.uncond:
            for i in range(0, 1, self.microbatch): # 这里的microbatch就是batchsize，超出了range则i就是0。
                tsize = batch[0][i : i + self.microbatch].to(dist_util.dev())# 12.21:这一句其实就相当于取batchsize了..
            # batch = batch.to(dist_util.dev())
        else:
            for i in range(0, 1, self.microbatch): # 这里的microbatch就是batchsize，超出了range则i就是0。
                tsize = batch[0][i : i + self.microbatch].to(dist_util.dev())
            # batch = batch.to(dist_util.dev())

        t, weights = self.schedule_sampler.sample(tsize.shape[0], dist_util.dev())
        # t = th.full((65536,), 15)
        # weights的数目和batchsize相同
        # 注意下面的functools partial，它可以固定一个方法的一部分参数，从而获得一个需要更少参数的新的可调用对象。第一个参数是需要partial的方法，后面是需要固定的参数。
        compute_losses = functools.partial(
            self.diffusion.training_losses, # ！！去深入地看一下代码中对这几种loss的综合处理和输出？？？！！
            self.ddp_model,
            batch,
            t,
            uncond = self.uncond
        )

        # if last_batch or not self.use_ddp: # ddp是一个分布式训练使用的模块
        #     # 如果是最后一个microbatch了或者不用ddp（# 那为什么last了useddp也不ddp了呢？）
        #     losses = compute_losses()  # 那就直接用上面固定了参数的compute_losses去计算loss（这里需要去看看loss的计算！！！！！）
        # else:
        #     with self.ddp_model.no_sync():
        losses = compute_losses() # 所以lastbatch是为了累积梯度的！自然如果不使用ddp的话，肯定就不计算了呗！
# 鬼鬼 似乎用上microbatch才好分布式训练？先试一下不开microbatch能不能多卡，不行再去看原代码里怎么做的。

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            ) # 如果设置了按loss采样t则调用这个

        loss = (losses["loss"] * weights).mean() # weights用来计算loss
        self.ldict.append(loss.item()) # 考虑优化一下loss保存？或者还是看看怎么把tmp路径改一下？

        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self): # Why calculate grad norm? What does this metric mean?
        sqsum = 0.0
        #print("master parameters:", i.is_leaf for i in master_params)
        #for i in self.master_params:
        #    print(i.is_leaf)
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self): # 此为学习率退火模块，如设置则学习率会按steps逐步下降。
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step) # logkv？
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt" # 现在应该是用这个存
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
                # with bf.BlobFile(bf.join("/workspace/codes/mediddpm_model_uncond_zinc_t_saved_50_lrup/", filename), "wb") as f:#
                #     th.save(state_dict, f) # mod.


        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir()) # os.environ.get（）是python中os模块获取环境变量的一个方法，
    # 可以设置默认值，当键存在时返回对应的值，不存在时，返回默认值，即从这里的环境变量中获得DIFFUSION_BLOB_LOGDIR的值，若不存在则直接用logger里的getdir的值


def find_resume_checkpoint(): # 这个是放在这里让你自己实现的一个方法，用于从自己的blob storage里寻找resume points.
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None: # 设定了checkpoint参数后应该这个main_checkpoint参数就不会是None了吧。这里需要看的是emacheckpoint的存储的是谁的参数呢？有什么不同呢？
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()): # 注意这里for循环里用到的zip方法来对两个数同时循环取
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
    # 这里存loss的形式是既存了各种loss，也储存了各个quartile的loss。每次run_step都会存一次的，存储的路径还是从getbloblogdir获得的，问题还是在这个log到的路径上。
