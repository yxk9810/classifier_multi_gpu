import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from .arguments import DataArguments


class TrainDatasetForCE(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
    ):
        self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        title = self.dataset[item]['title']
        content = self.dataset[item]['content']
        label = self.dataset[item]['label']
        item = self.tokenizer.encode_plus(title, content, max_length=self.args.max_len, truncation="only_second")
        item["label"] = label
        return item


class DevDatasetForCE(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
    ):
        self.dataset = datasets.load_dataset('json', data_files=args.dev_data,  split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        title = self.dataset[item]['title']
        content = self.dataset[item]['content']
        label = self.dataset[item]['label']
        item = self.tokenizer.encode_plus(title, content, max_length=self.args.max_len, truncation="only_second")
        item["label"] = label
        return item
