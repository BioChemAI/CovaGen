import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    这里对timestep的划分方式需要注意一下。
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):  # SpacedDiffusion类就是创建扩散模型的框架，注意重点实现在父类中
    # 为什么这个东西他叫SpacedDiffusion呢？：因为在iddpm论文中他提出的一个东西就是Spaced Timestep，是对timestep的一个优化
    """
    A diffusion process which can skip steps in a base diffusion process.
    对timestep做改进
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs): # init函数定义了加噪方案的β，timestep哪些时刻要保留，numstep加噪次数
        self.use_timesteps = set(use_timesteps) # 12.21： 如果使用respace的timestep，那么这里是经过过respace后的一系列timesteps
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"]) # β，也就是加的噪声，的数目，总是原steps个，所以这里通过取其长度获得了original num steps。

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        # 注意这里！上面进行的是，如果你repace你的steps，那么对β相应地进行respacing。
        super().__init__(**kwargs) # 用kwargs初始化父类

    def p_mean_variance( # p就是神经网络所预测的分布，故p_mean_variance就是神经网络预测的均值和方差
    # p_mean_variance主要是在sampling时会用到，同时如果使用klloss也会用到。
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
       return super().p_mean_variance(model, *args, **kwargs) #注意这里也修改了，没有wrapmodel
       # return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)
    # 12.18:因为要train所以又关掉了wrapmodel。wrapmodel影响的是t。
    # 3.15: 之前去掉了95行这里的wrapmodel，所以这里应该受影响的是sample过程的rescale

    def training_losses( # 就是之前根据传入的超参数不同得到不同目标函数loss的公式，最简单的就是MSE loss，我们也可以加上kl loss联合起来作为目标函数
            # 这个是对父类方法的一个重写，training losses就看父类中的主方法吧 !
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(model, *args, **kwargs) #注意这里修改了，没有wrapmodel。
        # 前面用partial，固定参数的目的就是在这儿
        #return super().training_losses(model, *args, **kwargs)

    def _wrap_model(self, model): #对timestep进行后处理，比如对timestep进行scale，对timestep进行一定的优化
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel: # 这个就是对t后处理用的 如上面的注释的那一句：scaling is done by the wrapped model
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map # 这个就是有序的respace后的steps
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype) # 把新steps转成一个tensor
        new_ts = map_tensor[ts]
        print("WRAP is really called") # 这里，就是在进行timestep的respace的mapping！
        # 就是，ts是直接采样得到的t们，就是你给出的respace后的t的个数。通过这里在前面创建的newt和originalt之间的mapping来把ts对应到对应的原先的steps上！
        # 意义何在？？？
        if self.rescale_timesteps:
            # rescale把timestep保持在0到1000之间，像原论文中那样。。
            print("Use rescaled timesteps")
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs) # 注意这种采用了**的不定长传参
