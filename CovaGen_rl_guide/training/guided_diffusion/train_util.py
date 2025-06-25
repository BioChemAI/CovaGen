import copy
import functools
import os
import xgboost as xgb
import pickle
from transvae.rnn_models import RNNAttn
from rdkit.Chem import AllChem
import blobfile as bf
import torch as th
import torch.distributed as dist#
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from .losses import gaussian_likelihood_vector,likelihood_for_prior,another_gaussian_likelihood_vector
from .fp16_util import unflatten_master_params
from rdkit import Chem
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .check_benzene import check_ben
from tqdm.auto import tqdm
from torch_geometric.data import Data,Batch

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = th.from_numpy(tensor)
    if th.cuda.is_available():
        return th.autograd.Variable(tensor).cuda()
    return th.autograd.Variable(tensor)

class TrainLoopRL:
    def __init__(
            self,
            *,  # 参数中星号是？
            model,
            prior,
            diffusion,
            data,
            batch_size,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            scoring_model=None,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            weight_decay=0.0,
            lr_anneal_steps=0,
            uncond=False,
    ):
        self.prior = prior  # 这是prior,创建的一个和agent相同的,不去更新参数的网络
        self.agent = model  # 这是agent,是train好的一个diffusion
        self.diffusion = diffusion  # diffusion框架\

        self.scoring_model = scoring_model

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
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        # self.mp_trainer = MixedPrecisionTrainer(
        #     model=self.model,
        #     use_fp16=self.use_fp16,
        #     fp16_scale_growth=fp16_scale_growth,
        # )

        self.uncond = uncond

        # 上面这个空行的含义应该就是，前面是传入的参数组成的属性，且基本不会再更改。后面是自带的属性，以及是调用对象时会被改变的属性？
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()  # global batch？ get_world_size？
        print("model's parameters:", model.parameters())
        self.agent_params = list(self.agent.parameters())
        self.prior_params = list(self.prior.parameters())
        self.master_params = self.agent_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()  # 这个参数用到哪儿了呢？sync_cuda
        self.ldict = []

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.cnt=0
        #载入xgb
        with open('/workspace/codes/emiddpm/toxity/xgboost_model_maccs.pkl', "rb") as model_file:
            self.xgb_model = pickle.load(model_file)
        self.vae = RNNAttn(load_fn='/workspace/codes/othercodes/vae_checkpoints/unchanged_smiles_inuse/checkpoints/080_NOCHANGE_evenhigherkl.ckpt')

    def _load_and_sync_parameters(self,model):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.agent.parameters())

    def predict_score_xgb(self,smi_list):
        molecules = [Chem.MolFromSmiles(smiles) for smiles in smi_list]
        maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in molecules if mol is not None]
        fp_array = np.array([list(fp) for fp in maccs_fps])
        dtest = xgb.DMatrix(fp_array)
        y_pred_loaded = self.xgb_model.predict(dtest)
        return y_pred_loaded

    def RLrun_mem(self,x,prev_x=None,t=None,log_prob=None,batch=None,prior_nll_func=None,rewards=None):
        '''
        :return:
        '''
        #现在所有的输入都是list
        epochs = 4
        accumulated_gradients = {name: th.zeros_like(param) for name, param in self.agent.named_parameters()}
        for epoch in range(epochs):
            for j in range(40):
                self._disable_prior_gradients()#

                # self.opt.zero_grad()
                initial_params = [param.clone() for param in self.agent.parameters()]

                log_prob_old = log_prob[j].to("cuda:0")
                prev_x_jth = prev_x[j].to("cuda:0")
                tth = t[j].to("cuda:0")
                sample = x[j].to("cuda:0")

                # NLL_agent_old = gaussian_likelihood_vector(sample,means=out["mean"],log_scales= 0.5 * out["log_variance"])
                # with th.no_grad():
                NLL_prior = prior_nll_func(self.agent,prev_x_jth,tth,sample=sample,batch=batch).detach()
                NLL_prior = -NLL_prior

                # calculate advantage from score
                advantages = (rewards - np.mean(rewards)) / np.std(rewards)
                ratio = np.exp(NLL_prior - log_prob_old)
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * np.clip(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
                loss = np.mean(np.maximum(unclipped_loss, clipped_loss))
                print(loss)
                # loss.requires_grad_(True)

                # Gradients.
                loss.backward()
                for name, param in self.agent.named_parameters():
                    accumulated_gradients[name] += param.grad.data
                self.opt.zero_grad()
                if tth.max()==0:
                    for name, param in self.agent.named_parameters():
                        param.grad.data = accumulated_gradients[name] / 200
                    self.opt.step()
                trained_params = list(self.agent.parameters())
                # print(self.prior.mlp.weight)
                parameters_updated = any((param1 != param2).any() for param1, param2 in zip(initial_params, trained_params))

                # train robustness check.
                if parameters_updated:
                    print("updated!")
                else:
                    print("nope,no update")
                # nlist = []
                # ylist = []
                # for name, param in self.agent.named_parameters():
                #     ylist.append(name)
                #     if param.grad is None:
                #         nlist.append(name)


        self.cnt+=1
        self.step+=0.05
        if self.cnt == 4000:
            self.save()
            self.cnt=0
        # prior怎么算likelihood？维护两条并行的采样？
        # 搞一个类似sampling，但不loop的prior计算。

    def split_into_four_parts(self,numb):
        # 确保数字是正数
        if numb <= 0:
            return "neg!"

        # 计算每个部分的大小
        part_size = numb / 4

        # 确保每个部分的大小是整数
        if part_size.is_integer():
            return [int(part_size)] * 4
        else:
            # 如果不能整除，则尝试以最接近的整数分割
            parts = [int(part_size)] * 3
            parts.append(numb - sum(parts))
            return parts

    def split_sample_batch(self,log_prob,prev_x,rewards,t,x):
        parts = self.split_into_four_parts(len(log_prob[0]))
        spl_log_prob, spl_prev_x, spl_rewards,spl_t,spl_x =[[] for _ in range(4)],[[] for _ in range(4)],[[] for _ in range(4)],[[] for _ in range(4)],[[] for _ in range(4)]
        for i in log_prob:
            result_tensors = th.split(i, parts)
            for i, sub_tensor in enumerate(result_tensors):
                spl_log_prob[i].append(sub_tensor)
        for i in prev_x:
            result_tensors = th.split(i, parts)
            for i, sub_tensor in enumerate(result_tensors):
                spl_prev_x[i].append(sub_tensor)
        for i in t:
            result_tensors = th.split(i, parts)
            for i, sub_tensor in enumerate(result_tensors):
                spl_t[i].append(sub_tensor)
        for i in x:
            result_tensors = th.split(i, parts)
            for i, sub_tensor in enumerate(result_tensors):
                spl_x[i].append(sub_tensor)
        result_reward = th.split(rewards,parts)
        for i, sub_tensor in enumerate(result_reward):
            spl_rewards[i].append(sub_tensor)
        return spl_log_prob,spl_prev_x,spl_rewards,spl_t,spl_x

    def split_data(data, split_sizes):
        # 确保总大小与分割大小列表相匹配
        assert sum(split_sizes) == data.num_nodes, "分割大小列表与数据大小不匹配"

        # 创建一个空列表用于存储分割后的 Data 对象
        split_data_list = []

        # 记录当前节点的索引
        node_index = 0

        # 循环遍历分割大小列表
        for size in split_sizes:
            # 提取节点、边和特征
            nodes = data.x[node_index: node_index + size]
            edges = data.edge_index[:, (data.edge_index[0] >= node_index) & (data.edge_index[0] < node_index + size)]
            edge_attr = data.edge_attr[(data.edge_index[0] >= node_index) & (data.edge_index[0] < node_index + size)]

            # 创建新的 Data 对象
            split_data = Data(x=nodes, edge_index=edges, edge_attr=edge_attr)

            # 更新节点索引
            node_index += size

            # 将新的 Data 对象添加到列表中
            split_data_list.append(split_data)

        return split_data_list

    def RLrun(self,x,prev_x=None,t=None,log_prob=None,batch=None,prior_nll_func=None,rewards=None):
        '''
        :return:
        '''
        epochs = 4
        # spl_log_prob,spl_prev_x,spl_rewards,spl_t,spl_x = self.split_sample_batch(log_prob,prev_x,rewards,t,x)
        accumulated_gradients = {name: th.zeros_like(param) for name, param in self.agent.named_parameters()}
        indices = tqdm(range(300))
        for epoch in range(epochs):
            loss_ls = []
            for j in range(300):#
                # self._disable_prior_gradients()#
                # self.opt.zero_grad()
                initial_params = [param.clone() for param in self.agent.parameters()]

                log_prob_old = log_prob[j].to("cuda:0")
                prev_x_jth = prev_x[j].to("cuda:0")
                tth = t[j].to("cuda:0")
                sample = x[j].to("cuda:0") # 老



                # NLL_agent_old = gaussian_likelihood_vector(sample,means=out["mean"],log_scales= 0.5 * out["log_variance"])
                # with th.no_grad():
                NLL_prior = prior_nll_func(self.agent,prev_x_jth,tth,sample=sample,batch=batch)
                # NLL_prior = -NLL_prior

                # calculate advantage from score
                # rewards = spl_rewards[epoch][0]
                advantages = ((rewards - th.mean(rewards)) / th.std(rewards)).to('cuda:0')
                ratio = th.exp(NLL_prior - log_prob_old)
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * th.clip(ratio, 1.0 - 1e-4, 1.0 + 1e-4)
                loss = th.mean(th.maximum(unclipped_loss, clipped_loss))
                # loss.requires_grad_(True)

                # Gradients.
                loss.backward()
                loss_ls.append(loss.item())
                for name, param in self.agent.named_parameters():
                    accumulated_gradients[name] += param.grad.data
                # self.opt.zero_grad()
                if tth.max()==0:
                    for name, param in self.agent.named_parameters():
                        param.grad.data = accumulated_gradients[name] / 300
                    self.opt.step()
                    self.opt.zero_grad()
                    for name in accumulated_gradients:
                        accumulated_gradients[name].zero_()
                    print('mean loss', np.mean(np.array(loss_ls)))
                self.opt.zero_grad()

                # nlist = []
                # ylist = []
                # for name, param in self.agent.named_parameters():
                #     ylist.append(name)
                #     if param.grad is None:
                #         nlist.append(name)


        self.cnt+=1
        self.step+=0.05
        if self.cnt == 15:
            self.save()
            self.cnt=0
        # prior怎么算likelihood？维护两条并行的采样？
        # 搞一个类似sampling，但不loop的prior计算。



    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        for param in self.prior_params:
            param.requires_grad = False


    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{({round(self.step, 2)}+self.resume_step)}.pt"
                else:
                    filename = f"ema_{rate}_{({round(self.step, 2)}+self.resume_step)}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        # for rate, params in zip(self.ema_rate, self.ema_params):
        #     save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step)}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()
    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.agent.parameters(), master_params
            )
        state_dict = self.agent.state_dict()
        for i, (name, _value) in enumerate(self.agent.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

class TrainLoop:
    def __init__(
        self,
        *,
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
        scoring_comp = None
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.scoring_model = scoring_comp
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
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch = next(self.data)
            batch = batch.to(dist_util.dev())
            self.run_step(batch)
            # if self.step % self.log_interval == 0:
            #     logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch,):
        self.forward_backward(batch)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        # for i in range(0, batch.shape[0], self.microbatch):
        #     micro = batch[i : i + self.microbatch].to(dist_util.dev())
        #     last_batch = (i + self.microbatch) >= batch.shape[0]
        t, weights = self.schedule_sampler.sample(128, dist_util.dev())
        # t = tensor(t[0],200)
        criterion = th.nn.MSELoss()
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            batch,
            t,
            scoring=self.scoring_model,
            crit=criterion,
        )

        loss = compute_losses()

        # if isinstance(self.schedule_sampler, LossAwareSampler):
        #     self.schedule_sampler.update_with_local_losses(
        #         t, losses.detach()
        #     )
        #
        # loss = (losses["loss"] * weights).mean()
        print(loss)
        # for param in self.ddp_model.parameters():
        #     if param.grad is not None:
        #         print(param.grad.max())
        # log_loss_dict(
        #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
        # )
        self.mp_trainer.backward(loss)
        for param in self.ddp_model.parameters():
            if param.grad is not None:
                print(param.grad.max())

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
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

    def RLrun(self,x,out,t=None,batch=None,prior_nll_func=None):
        '''
        :return:
        '''
        # 在这里，接受每步得到的μ，σ和xt，用它们更新模型的参数，在200steps（暂定）更新完成后，进入下一个batch？
        # 也就是说，SampleLoop正常进行，在其后加这个RLLoop以更新参数
        # TODO 是在200step后更新prior，还是在每一步都去更新prior？
        self._disable_prior_gradients()#
        sample = x # 现在可以保证的是，这是用agent获得的sample
        # NLL_agent = gaussian_likelihood_vector(sample,means=out["mean"],log_scales= 0.5 * out["log_variance"])
        # with th.no_grad():
        #     NLL_prior = prior_nll_func(self.prior,x,t,batch=batch)
        mscore = self.scoring_model.predict(sample,t)# TODO:如果在这里加decoding呢？
        mscore.requires_grad_(True)

        # augmented_likelihood = NLL_prior - 10 * to_tensor(mscore)
        # loss = th.pow((augmented_likelihood - NLL_agent), 2)
        loss = 10*to_tensor(mscore)
        loss = loss.mean()
        print(loss)
        loss.requires_grad_(True)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # for param in self.agent.parameters():
        #     if param.grad is not None:
        #         print(param.grad.max())
        self.cnt+=1
        self.step+=0.05
        if self.cnt == 8000:
            self.save()
            self.cnt=0

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
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
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
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
