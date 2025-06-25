from contextlib import contextmanager
import datetime
from itertools import combinations
import json
from pathlib import Path
import logging
import secrets
import tempfile
import toml
import torch
import random
import numpy as np
from typing import Generator, Hashable, Union, List
StrPath = Union[str, Path]



def use_path(dir_path: StrPath = None, file_path: StrPath = None, new: bool = True) -> Path:

    assert sum((dir_path is None, file_path is None)) == 1, \
        Exception(f"There is only one in dir_path ({dir_path}) and file_path ({file_path})")

    if dir_path is not None:
        _dir_path = Path(dir_path)
        if new:
            _dir_path.mkdir(exist_ok=True, parents=True)
        else:
            assert _dir_path.exists(), Exception(f"{_dir_path} does not exist")
        return _dir_path

    if file_path is not None:
        _file_path = Path(file_path)
        if new:
            _file_path.parent.mkdir(exist_ok=True, parents=True)
        else:
            assert _file_path.exists(), Exception(f"{_file_path} does not exist")
        return _file_path

def init_logging(log_fn: Union[StrPath, List[StrPath]] = None):
    log_fns = log_fn if isinstance(log_fn, list) else [log_fn] if log_fn is not None else []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s,%(lineno)d,%(funcName)s | %(message)s",
        handlers=[
            *(logging.FileHandler(x) for x in log_fns),
            logging.StreamHandler()
        ]
    )

@contextmanager
def get_temp_fn(data: bytes = None, temp_dir: StrPath = None, suffix: str = None, 
    prefix: str = "rFragLi-") -> Generator[Path, None, None]:

    f = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=suffix, prefix=prefix)
    if data is not None: f.write(data)
    f.close()
    try:
        yield Path(f.name)
    except Exception as error:
        raise error
    finally:
        Path(f.name).unlink()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def find_by_key_value(lst, key, value):
    return next((x for x in lst if x[key] == value), None)

# The upper one is very slow on 160000 data sets, and the next one is light speed
def find_by_key_value_cached(lst, key):
    value_to_item_dict = dict((x[key], x) for x in lst)
    def f(value):
        return value_to_item_dict.get(value, None)
    return f

def get_time_str():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def generate_train_id():
    time_str = get_time_str()
    token_hex = secrets.token_hex(2)
    train_id = f'{time_str}_{token_hex}'
    return train_id

def init_random_seed(random_seed):
    r = random.Random(random_seed)
    random.seed(int(r.random() * (1<<32)))
    np.random.seed(int(r.random() * (1<<32)))
    torch.manual_seed(int(r.random() * (1<<32)))

def parse_multi_toml(fns: List[StrPath]):
    dic = dict()
    for fn in fns:
        with open(fn, 'r') as f:
            dic = {**dic, **toml.load(f)}
    return dic

def statistics_to_str(arr: np.ndarray, name: str):
    return f"{name}_len_min_max_mean_median: {len(arr)}, {np.min(arr)}, {np.max(arr)}, {np.mean(arr):.3f}, {np.median(arr):.3f}"

def choose_weight(best_fn: Path, weights_dn: Path, weight_choice: str):

    # 1. Get directly from the best file ["recons", "kld", "all"]
    with open(best_fn, 'r') as f:
        best_dict = json.load(f)
    if weight_choice in best_dict["model"].keys():
        return Path(best_dict["model"][weight_choice])

    weight_list = []
    for _fn in weights_dn.glob("*.pt"):
        _prefix = _fn.name.split('_')[0]
        try:
            _epoch = int( _prefix)
        except ValueError:
            _epoch = -1
        weight_list.append(dict(epoch=_epoch, prefix=_prefix, fn=_fn))

    # 2. "latest"
    if weight_choice == "latest":
        return max(weight_list, key=lambda x: x["epoch"])["fn"]
    
    # 3. Specify prefix (epoch number) ["023", "033", ...]
    weight_fn = find_by_key_value(weight_list, "prefix", weight_choice)
    if weight_fn is not None:
        return weight_fn

    raise Exception(f"Unable to determine weight: {weight_choice}")

def mean_group_similarity(lst: List[List[Hashable]]):
    sim_list = []
    for x, y in combinations(lst, 2):
        m = x + y
        n = set(x) & set(y)
        sim = sum(m.count(i) for i in n) / len(m)
        sim_list.append(sim)
    return sum(sim_list) / len(sim_list)

def mean_without_none(lst):
    return np.mean([x for x in lst if x is not None])

def mean_without_empty(lst):
    return np.mean([x for x in lst if len(x)])
