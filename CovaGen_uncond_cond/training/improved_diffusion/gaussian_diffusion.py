"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum # 这是python自带的用于实现枚举的数据类型的模块，改进了如果使用普通的类来实现枚举的局限性。
from datetime import datetime
from torchviz import make_dot
# 枚举的类型一般有两个特点：1.key不应相同. 2.应不能在外部进行修改
import math

import numpy as np
import torch as th
import pickle
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

def steps_save():

    with open("/mnt/ldata1/save_by_steps/") as f:
        print("done")

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        print("using LINEAR noise schedule")
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine": # 这
        print("using COSINE noise schedule")
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    # 用于公式16后的βt的计算
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        # 这里控制了beta的上界（是不是这样子就留下了一点点的x0在纯gauss噪声中？）
    return np.array(betas)


class ModelMeanType(enum.Enum): # 这种用法就是对enum枚举的标准创建：继承enum！
    """
    Which type of output the model predicts.
    """
    # 一些场景下 我们并不直接关心枚举成员的值是什么，这时可以用.auto()来处理其值
    # 枚举的成员有两个内置属性，name和value，还可以通过其他方式添加其他属性。
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    训练和抽样扩散模型的实用程序，找了半天，原来这里才是真正的实现类。。
    这个就是最原来的DDPM的一个代码实现
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas, # 就是β，每步加噪的β值。记得在子类里通过cumprod实现的β不还
        model_mean_type, # 知道这个模型要预测什么，预测的是方差还是噪声还是x0（？）
        model_var_type, # 固定方差还是学习方差，抑或是学习线性加权的权重
        loss_type, # loss是预测mse还是加kl
        rescale_timesteps=False, # 对时间进行scale，使得timestep永远缩放到在0到1000之间(?)
            # 即即使你设置timestep是300，我们仍然可以把他scale到0-1000，这个是和原论文对应的。
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type # 方差是固定还是学
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
# 前面这四行就是在决定这个模型的mean在预测什么，是噪声还是x0还是xt-1的均值，以及方差做什么。
        # Use float64 for accuracy.
        # 接下来是 前向过程 的一些关键量
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D" # betas不是1d则报错
        assert (betas > 0).all() and (betas <= 1).all() # 确保β都在0到1之间（这个β的数目理应和扩散步数一样吧？）

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas # 这个就是α
        self.alphas_cumprod = np.cumprod(alphas, axis=0) # α-bar，就是连乘呀
        # prod函数用于求矩阵元素的积，其调用格式如下。
        # （1）B=prod(A)：若A为向量，则返回所有元素的积；若A为矩阵，则返回各列所有元素的积。
        # （2）B=prod(A,dim)：返回矩阵A中的第dim维方向的所有元素的积。
        # 对数组计算累积连乘。如果A是一个向量，函数返回一个长度相同的向量，其中的元素是原向量的累积连乘。
        # 如果A是一个矩阵，则将每一列当做一个向量进行计算，最后返回与A大小相同的矩阵。
        # 如果A是一个多维数组，函数对第一个长度不为1的维度进行计算。
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # αt-1 bar
        # 一般做cumprod的第0项我们都用1.0去填充？（意味着后面这个cumprod向量是1：项）
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0) # αt+1 bar
        # 注意t+1这一列中的最后一项用0。
        # 这真是巧妙的和α和β的长度对应关系。α和αt-1、αt+1和β形状都是一样的（当然肯定是得一样的，毕竟扩散过程t步是确定的）
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,) # 注意这一句！形状不一样了，直接报错了就

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 接下来计算扩散过程中后验分布的真实的方差和均值。方差可以直接计算，而均值和xt有关
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) # 根号下αt-bar
        # 用于算后验分布μ，或者由x0算任意xt
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) # 根号下1-αtbar
        # 用于公式8的标准差计算
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)  #log 1-atbar
        # 用于求loss
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod) # 根号下 αtbar分之1 recip指倒数
        # 会用在算μ中
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1) #根号下 αtbar-1分之一（这用在？）
        #m1 指minus1。。

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ) # 这个就是β~t， 为什么说是后验分布的 真实 方差？
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) # 这里做了一手截断，因为怕posterior_variance第一项，也就是扩散链刚开始时它为0。
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        ) # 这里这俩 coef1和coef2， 是μ~t（xt，x0）的两个系数！！。。
#到这里 这里的init函数就把一些确定的变量给算好了
#3.15: 注意，一切的这些量都是从最一开始的beta β算出来的，因此一切这些量的形状都是beta的形状，beta是原公式里的加噪，而beta是在create diffusion时被确定的，它的形状
#就取决于选择的timesteps有多少步，比如100步，那么beta就是一个长度为100的1d array！


    def q_mean_variance(self, x_start, t): # q通常指真实的分布
        """
        Get the distribution q(x_t | x_0). 这个是算公式8q（xt，x0）用的

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None): # 给定x0和步数t直接算出xt，也就是说q_sample用于获得加噪的数据
        # 用于加噪出xt，其实是一个利用了重参数的过程（重新去看看重参数！？）
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0). 哦哦，就相当于从给定x0算xt的分布中采样了。

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        # 基于x0，xt算出后验分布中真实的均值和方差。即算出公式9-11！
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """

        #print("---now we're in q post m v---")
        #print("x start shape:",x_start.shape)
        #print("x t shape:", x_t.shape)
        #x_t = x_t.unsqueeze(0) sample时用
        assert x_start.shape == x_t.shape # 这里报错是因为sampling我已经改过了，导致train和samp这里的x_start并不一样。讲道理处理一下把batch中的target拿出来就可以的，但是还有后续呢
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )# 此为公式11
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def  p_mean_variance(
        self, model, x, t, batch = None, clip_denoised=True, denoised_fn=None, model_kwargs=None,look=0,num_samples=None,fresh_noise=None,same_start=False
    ):
    # 这里到了p分布。p分布是神经网络的分布，去建模拟合的分布，
    # 这里能得到前一时刻（逆扩散过程）的均值和方差，也包括x0的预测@
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        dts = batch
        # x1 = x
        #print("p_mean_variance jinlaile?", dts)
        
        B, C = x.shape[:2]
        #assert t.shape == (B,)
        #print("zuihou x",x)
        #model_output = model(dts, x, samp = True)
        #print("x de shape:",x.shape)
        #print("t de shape:",t.shape)
        if same_start: # mod.
            t=t.repeat(10) # mod.
            if look==0:
                x = x.repeat(10,1)
                print("Using Same Start")
        # print("rpt x shape:",x.shape)#
        if dts!=None:
          esm_cond = batch[0].to("cuda:0")
          mask = batch[1].to("cuda:0")
          model_output = model(t, esm_cond=esm_cond, mask=mask, noidata=x) # 这里x的逻辑应该没问题
        elif dts==None:
          model_output = model(t,noidata=x)
        time_1 = datetime.now()
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:]) # 需要去看原先x的2：后的shape对应的到底是什么
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else: # 不学方差
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart( # 步：是不是得去掉这个process的clip？？？
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)
        
        #print("---now back in p m v---")
        #("modelmean shape:",model_mean.shape)
        #print("model_log_variance shape:",model_log_variance.shape)
        #print("pred_xstart.shape:",pred_xstart.shape)
        #print("x shape",x.shape)
        #model_mean = model_mean.squeeze(0)
        #pred_xstart = pred_xstart.squeeze(0)
        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        time_2=datetime.now()
        #print("time spend on other calculation:", (time_2-time_1).microseconds)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            # "origX": x1
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
    # 用于给定t处的噪声来预测x0，即给定xt，t和x0到xt所加的噪声反推出x0
        #print("x_t shape:", x_t.shape)
        #print("eps shape:", eps.shape)
        #x_t = x_t.unsqueeze(0)
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
    # xt-1就是μ~t，由xt，反推出x0
    # 这个是公式10
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
    # 从预测出x0和xt，推导eps，也就公式8，也就是预测x0到xt的噪声
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample( # 这个是在推理时会用到
        self, model, x, t, batch = None, clip_denoised=True, denoised_fn=None, model_kwargs=None,look=0,num_samples=None,fresh_noise=None,same_start=False
    ):
        #从xt采样出xt-1，所有的p分布都是模型预测的，其实就是推理的函数！！
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        time_1 = datetime.now()
        dts = batch
        #print("psample laile?",dts)
        out = self.p_mean_variance(
            model,
            x, # 这里传进来的x是每步的img
            t,
            fresh_noise=fresh_noise,
            batch = dts,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            look=look,
            num_samples=num_samples,
            same_start=same_start,
        )

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        #print("outmeanshape:",out["mean"].shape)
        #print("zeromask:",nonzero_mask.shape)
        #print("lvariance:",out["log_variance"].shape) # logvariance是方差，这里是固定的，max是β，min是β~
        #print("noise.shape:",noise.shape)
        if same_start:
            nonzero_mask=nonzero_mask.repeat(100,1) # mod.这个repeat居然没啥问题
        if look ==0 and same_start: # mod. 此处是在初始时把noise给重复，后面形状自然就相同，不必重复
            noise = noise.repeat(10,1)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise # 步：是不是这个log variance和mean有一些不同的操作？
        # 要看一下这对应这哪个的公式
        time_2 = datetime.now()
        #print("time spent for 1 pmeanvariance and one sample", (time_2 - time_1).microseconds)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}#, "origx": out['origX']}

    def p_sample_loop( # 把上面的psample循环一下，就是串起来了所有的步骤，生成最终的样本
        self,
        model,
        shape,
        batch = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        onestep=False,
        num_samples=None,
        same_start=False# arg to control the amout of samples generated for each protein. actually controls the time repeated of two conditions in models.
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.#
        # 3.16 changelog: Add and argument to the denoising transition, the original sampled noise, to use it as the original
        # target just like the training procedure, so that it could be used to minus output from, which in a word is a change
        # to the model's output. In Dynamic_samp Before: error = noidata - output. After: error = orig_noise_saved - output
        # After trying: Failed. Seems that the original realization is correct.
        """
        final = None
        for sample in self.p_sample_loop_progressive( # 还是调用了下面定义的这个方法
            model,
            shape,
            batch,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            onestep=onestep,
            num_samples=num_samples,
            same_start=same_start
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        batch = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        onestep=False,
        rep_noise=False,
        num_samples=None,
        same_start=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        dts = batch # 此为全部的数据，100个pocket
        #print("chuanjinlaile?:",dts)
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        onestepvecs = {}
        onestepvecs[f"{self.num_timesteps}"] = img
        fresh_noise=img
        #print("fresh img:",img)
        indices = list(range(self.num_timesteps))[::-1]
        # indices就是时刻！：：表示对列表取一个逆序，这样就在反向地推理了！
        #print("progress?",progress)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        ky=0 #mod.
        # print("shape0:",shape[0])
        for i in indices: # 这个循环就是timesteps 的循环。以1000个去噪steps为例，前面的indices生成了从1000到0的steps，这个循环就一个个取出来。
            #也就是说，前面传入的（100，128）这里实际是batchsize*vectorsize 的形式，而这个batch中每个向量都对应着不同的pocket（这里去看一下？）
            t = th.tensor([i] * shape[0], device=device)
            #print("fresh t:",t) # 这就是DDPM论文算法部分的sample算法的t的走向。
            with th.no_grad():
                out = self.p_sample( # img代表每次推断一步的结果。
                    model,
                    img,
                    t, # 是 [100]的形状
                    fresh_noise=fresh_noise,
                    batch = dts,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn, # unused
                    model_kwargs=model_kwargs,
                    look=ky,
                    num_samples=num_samples,
                    same_start=same_start,
                )
                yield out
                img = out["sample"]
                # img = out["origx"]
                if i!=0:
                    if ((i+1)%100)==0: # when sampling many steps, here chooses every 100 steps to save.
                        onestepvecs[f"{i}"] = img
                elif i == 0:
                    onestepvecs[f"{i}"] = img
            ky+=1
        if onestep == True:
            with open("/workspace/codeeval/save_by_steps/onesteps_250k3ksteps_1proteinUPPS_normalnoise_linear.pkl", "wb") as f:
                pickle.dump(onestepvecs, f)

# Down below is the implementation for DDIM, haven't checked.？
# According to the course the DDIM traded speed with the diversity of the data denoised.
    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd( # 这个是计算loss了，bpd是bit per dimension 这个很重要！
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        # kl散度包括两项，当t在0到t之间，用模型预测分布计算高斯分布算一个kl散度，另一项是最后一个时刻，L0 loss，
        # 使用的是似然函数，负对数似然函数，使用的是累积分布函数的差分拟合离散的高斯分布
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # 真实的x0，xt和t去计算出xt-1的均值和方差
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        # xt、t和预测的x0去计算出xt-1的均值和方差
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        # 尔后就是pθ和q分布之间的kl散度！注意这里！就是用的上面的计算出的
        # 这里对应着原始损失函数L中的Lt-1
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)
        # 这里对应着L0，用的负对数似然，是累计分布函数的差分来拟合离散的高斯分布？？？
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # t=0用离散的高斯去计算似然，t>)直接用KL
        output = th.where((t == 0), decoder_nll, kl) # 注意这里where的使用
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, batch, t, model_kwargs=None, noise=None,uncond=False): # 实际上此方法算是整个DM的入口
        """
        用这个函数来确定这个模型到底有哪些loss。注意它是
        三种情况：只学习vbloss，只学习MSEloss，同时学习MSE和vb loss(vb?)
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if not uncond:
            bs = len(batch[0]) # 额就是batchsize...
            if model_kwargs is None:
                model_kwargs = {}  # 原micro_cond的参数，应该是用不到了。
            if noise is None:
                noise = th.randn_like(batch[2].view(bs, -1)).cuda()
            retarget = batch[2].view(bs, -1).cuda()
            esm_emb = batch[0].cuda()
            mask = batch[1].cuda()
            # bt = x_start.batch
        else:
            bs = x_start.shape[0]
            if model_kwargs is None:
                model_kwargs = {}   # 原micro_cond的参数，应该是用不到了。
            if noise is None:
                noise = th.randn_like(x_start)   # 这个整个摊平是iddpm生成噪声的方式，还是是我加的？
            #noise = th.randn_like(x_start)
        #x_t = self.q_sample(x_start.target, t, noise=noise) # 即取得len(t)个分别是t中元素步的被diffuse的x。这是一个list，里面有一系列不同t的diffuse后的x。
            retarget = x_start
        x_t = self.q_sample(retarget, t, noise=noise)
        # 这是重点的一步 是通过这个方法取得了我需要的target！

        terms = {} # terms？：存loss项用


        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            #可以直接是kl
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=batch,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE: # 来，我们就先只用MSE吧。。
            #也可以是mse和kl的结合
            #model_output = model(x_t, self._scale_timesteps(t), **model_kwargs) # 重点！！！！！！终于用到了model！
            if not uncond:
                model_output = model(t, esm_cond=esm_emb, mask=mask, noidata=x_t)
            else:
                model_output = model(t, noidata=x_t,batch=bs)

            # dot = make_dot(model_output, params=dict(model.named_parameters()))
            # dot.format = 'pdf'
            # dot.render("model_graph")#
            # x_t是len(t)个diffusion了的样本；_scale_timesteps(t)？（是不是前面提到的给timestep一个scale化？）；model_kwargs中就一个micro_cond（去查查对这种参数取**是什么效果？）
            #model_output = th.flatten(model_output)
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]: # 如果学习方差的话那么进行下面的操作调整loss
            # 要把学习方差调整出来的话，这里的BC的形状得好好理一理了，前面为了repeated sampling我去掉过一个BC的形状判定了已经，一定注意！！？？？！
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1) # 原先的model输出是两个，含有对方差的预测。去看一下原代码的实现，想想在这个模型中对方差的学习是怎么实现的？
                # 但是感觉还是得去看原文的公式和方法才能整明白这里学习方差到底是个什么方式？
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                # 以变分下界学习方差，但是不要让其影响我们对均值（也就是数据本身）的预测
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1) # detach？torch中这几个方法里面的dim参数是指？
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE: # 是说，不能让loss中对方差的预测的那个项，伤害到mseloss这一项，所以用这一项乘timestep/1000。
                    # 还是去看原来的方法的对这个方差项还做了什么处理吧。。去看iddpm论文咯？？
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = { # 注意target这一部分，很重要，它决定了最后拿什么跟model_output算loss。
                # ！！他就是所说的你模型预测的是什么来算loss！这里ModelMeanType.EPSILON: noise就指模型预测的是noise，和noise算loss！
                # ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                #     x_start=x_start.target, x_t=x_t, t=t
                # )[0],
                ModelMeanType.START_X: batch[2],
                ModelMeanType.EPSILON: noise,#
            }[self.model_mean_type]#
            # ！ 注意了，这里用不同类型的输出数据来计算loss也可以是一个超参数！？？！记得把这个也实现了，我认为现在要把这个实现了得调整x_start.target的形状了。
            #print("model_output.shape:",model_output.shape)
            #print("target.shape",target.shape)
            #print("x_start.target.shape",x_start.target.shape)
            assert model_output.shape == target.shape #== x_start.target.shape
            terms["mse"] = mean_flat((target - model_output) ** 2) # 注意这一句！这就是loss的最终的计算！用target减去上面经过model得到的model_output的平方，就是论文中最原始的loss！
            if "vb" in terms: # 如果terms里有vb
                terms["loss"] = terms["mse"] + terms["vb"]
            else: # 抑或是纯的mseloss，我就先用这个了，纯的mseloss，terms这个dict里只有一个item，loss：mse的value。
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        # 这是一个先验的KL，它不影响模型的训练，它不含参（那他算出来是？）
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        # 从T到0的每个loss都算出来，这个是用来eval的，训练时用不到
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    # 一个辅助函数，从这个tensor中取第几个时刻。
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    # 3.15 拿betas为例，arr就是betas那个长度为timesteps的序列，注意，timesteps是一个一维的array，长度是batch的大小！
    #拿sampling为例，timesteps的长度就是同时生成多少个vector的数目!
    # 所以说，979这一句的含义就是:首先将arr转为tensor，再从中取出[timesteps]长度的对应timesteps位置的量！所以是res是10000的长度！
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None] # 这里是增加维度。
    return res.expand(broadcast_shape) # 重新去看一下np和torch的boardcast！？和expand方法！
