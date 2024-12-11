import os.path as osp
from os import environ

from datasets import Dataset, concatenate_datasets, load_dataset

from opencompass.registry import LOAD_DATASET
# from opencompass.utils import get_data_path

from .base import BaseDataset

@LOAD_DATASET.register_module()
class CMath(BaseDataset):
    def load(name: str, path: str, *args, **kwargs):
        dataset = load_dataset("weitianwen/cmath")
        return dataset
