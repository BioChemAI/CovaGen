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
     # 创建argumentParser，遍历字典添加到argparser中，这样省的我们一个个去写手写add_argument，是一个很好的可以学习的简洁写法
    dist_util.setup_dist() # 这一句用于分布式训练
    logger.configure() # 回去好好学一遍logger！！ 至少搞清楚logger什么时候输出到控制台、输出什么、各个常用log命令的用法、如何和何时把log文件保存
    logger.log("creating model and diffusion...")
    _1, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    ) # 这里创建model，得到UNet model 和diffusion框架
    # args是从命令行得到的一个很大的参数集，他其中包括了model、diffusion，也包括了训练的各种超参数
    # 所以这里采用argstodict方法，传入 model_and_diffusion_defaults().keys()，只取出model和diffusion需要的参数。
    model = opmd.Dynamics_revive(condition_dim=28, target_dim=128, hid_dim=64, condition_layer=3, n_heads=2, condition_time=True)
    model.to(dist_util.dev()) # 模型复制到GPU
    # model.train()e
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
# 返回的是一个采样器，可以是uniform，或者是基于loss重要性采样(二阶动量平滑loss，loss-second-moment？)
    #12.20：这个采样器部分需要去重新看一下并修改一下代码，如果是基于loss采样t，那么会有一个weight存在，看一下原代码中是怎么实现结合这个weight的
    # 现在只有均匀采样可用。
# 采样器的作用是：就是train算法里的那个时刻t的采样方式。
    logger.log("creating data loader...")
    data = load_data_smi(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        vae_dir=args.vae_dir,
        dataset_save_path=args.dataset_save_path,
        data_mood="train"
    ) # loaddata方法递归地找到所有的图片文件，对datadir下的都遍历
    #注意，loaddata是个yield生成器，后面在运行时有一个next方法在取之 train_util:116: batch, cond = next(self.data) # 从data迭代器中一个个地
#class_cond，类别判断，找到图片的每个类别，假设文件名的下划线的第一部分就是类别，用split做分割，将class排序设置索引，最终模型输出的还是索引
    logger.log("training...") # 是不是就是因为logger会自动维护一个logger使得你可以把控制台信息写入文件？
    # 最后实例化Trainloop类并直接调用其run_loop方法，把data，model，diffusion送入，开始训练。
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size, # done
        microbatch=args.microbatch, # done
        lr=args.lr, # adjust
        ema_rate=args.ema_rate, # adjust # 指数下降平均，似乎和梯度下降优化有关，这个还不够理解 去看？！！
        log_interval=args.log_interval, # done
        save_interval=args.save_interval, # done
        resume_checkpoint=args.resume_checkpoint, # done
        use_fp16=args.use_fp16, # done
        fp16_scale_growth=args.fp16_scale_growth, # done
        schedule_sampler=schedule_sampler, # !adjust loss sampler还未用上。？
        weight_decay=args.weight_decay, # 还未用上？
        lr_anneal_steps=args.lr_anneal_steps, # 还未用上？
    ).run_loop()
# 总体来说：
# 整个训练框架分为三步，第一步超参数汇总生成argparser，第二步create model and diffusion，第三步trainloop开始训练

def create_argparser():
    '''自动地从字典中生成命令行传参的argumentparser，免去了手打的反复'''
    defaults = dict( # 一系列超参。
        data_dir="/workspace/datas/11/dataset/crossdocked_pocket10",
        vae_dir="/workspace/codes/vaemodel/070_rnnattn-256_zinc.ckpt",
        dataset_save_path = "/workspace/datas/11/dataset/processed",
        schedule_sampler="loss-second-moment", # 嗯？这个和noise-schedule两个参数的区别是？
        lr=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=256,
        microbatch=-1,  # -1 disables microbatches。microbatches 是完全用不上了，想想怎么去掉？
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=1000,
        resume_checkpoint="", # 以后这个还是从命令行传进去为好？这些默认参数应该会被命令行覆盖掉吧？
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults()) #后面括号中的函数返回一个字典
    #  update()方法用于更新字典中的键/值对，可以修改存在的键对应的值，也可以添加新的键/值对到字典中。
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    #这个方法用于把刚才扩充的字典的所有参数都添加到parser中去（所以重点还是得理解parser）
    # 就是自动地把所有的命令行参数的add_parser都写好了，这个方法很聪明！！
    return parser # 随后就返回了总的命令行参数


if __name__ == "__main__":
    main()
