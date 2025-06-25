from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from optdiffusion import crossdock_dataset
from torch_geometric.data import DataLoader
from torch.utils.data import Subset

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    #通过 _list_image_files_recursively这个方法来找到datafir路径下的所有图片
    classes = None
    if class_cond: # 如果需要条件生成，假设文件名的第一部分就是类别名（）
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        # 相当于nlp中给一个个的token序号化，因为毕竟送到网络里需要是整形的id。（注意这里！）
        classes = [sorted_classes[x] for x in class_names]
    # 接下来，用前面的数据和信息创建一个ImageDataset的对象，然后把这个对象dataset传入dataloader就得到了dataloader。
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def split(dataset,split_file):
    split_by_name = torch.load(split_file)
    split = {
        k: [dataset.name2id[n] for n in names if n in dataset.name2id]
        for k, names in split_by_name.items()
    }
    subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
    return dataset, subsets

def load_data_smi( *, data_dir, batch_size, image_size = None, class_cond=False, deterministic=False, data_mood = "Train"):
    # 用了自定义的split方法，现在此方法应该是只对CrossDock数据集管用
    dataset = crossdock_dataset.PocketLigandPairDataset(
        '/workspace/stu/ltx/nt/dataset/dataset/',  # 这是 raw path
        vae_path='/workspace/stu/ltx/nt/045_trans1x-128_zinc.ckpt',
        save_path='/workspace/stu/ltx/nt/dataset/dataset/processed'
    )
    datafull, subsets = split(dataset,'/workspace/stu/ltx/nt/dataset/dataset/crossdocked_pocket10/split_by_name.pt')
    #if deterministic:
    #    loader = DataLoader(
    #        dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #    )
    #else:
    train, val = subsets['train'], subsets['test']
    print(f"Number of Training Data:{len(train)}")
    print(f"Number of Test Data:{len(val)}")
    follow_batch = ['protein_pos', 'ligand_pos'] # 还是有必要去再看看follow_batch的用法的。这个应该是pyg的Dataloader的？

    if data_mood =="train":
        loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, follow_batch=follow_batch
        )
        while True:
            yield from loader
    elif data_mood =="sample":
        loader = DataLoader(
            train, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True, follow_batch=follow_batch
        )
        return loader # 这个warning具体什么含义呢？

def load_data_smi2( *, data_mood, batch_size, image_size = None, class_cond=False, deterministic=False):
    # 修改的load_data用于创建PLPds这个类的dataset
    dataset = crossdock_dataset.PocketLigandPairDataset(
        '/workspace/stu/ltx/nt/dataset/dataset/',  # 这是 raw path
        vae_path='/workspace/stu/ltx/nt/045_trans1x-128_zinc.ckpt',
        save_path='/workspace/stu/ltx/nt/dataset/dataset/processed'
    )
    datafull, subsets = split(dataset,'/workspace/stu/ltx/nt/dataset/dataset/crossdocked_pocket10/split_by_name.pt')
    #if deterministic:
    #    loader = DataLoader(
    #        dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #    )
    #else:
    train, val = subsets['train'], subsets['test']
    print(len(dataset), len(train), len(val))
    follow_batch = ['protein_pos', 'ligand_pos']
    if data_mood == "train":
        loader = DataLoader(
                train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, follow_batch = follow_batch
            )
        print("Training")
    elif data_mood == "val":
        loader = DataLoader(
                val, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, follow_batch = follow_batch
        )
        print("Testing")
    while True:
        yield from loader # 注意这里可能会出现问题。
    #return loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
    # init可以找到所有的图片路径和类别
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
# getitem传入index获取每张图片，进行处理获取单张的训练样本，图像处理进行resize，转换RGB格式，归一化到-1到1之间的浮点型
# 注意他这种getitem的写法，他对要取出的数据做了很多的操作

# 注意！getitem每次传入一个idx（idx是不是就是由dataloader的一些策略来forge出的？），都能返回一个处理好的样本。
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
    # 返回一个图像的array，一个一个outdict储存图像的类别信息。
