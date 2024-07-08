# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
import sys
import os.path as osp
from itertools import chain
from pathlib import Path
import pickle
from typing import Any, List, Union
import subprocess
import mmap

import numpy as np
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from ttt.dataloader.lm_dataset import RandomFaultTolerantSampler, LMDataset
from ttt.infra.jax_utils import master_print


class LMDataModule:
    def __init__(
        self,
        dataset_name,
        tokenizer_name,
        dataset_config_name=None,
        max_length=1024,
        cache_dir=None,
        raw_json_path=None,
        val_ratio=0.0005,
        val_split_seed=2357,
        add_eos=True,
        batch_size=32,
        batch_size_eval=None,
        num_workers=1,
        loader_workers=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        fault_tolerant=False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.raw_json_path = raw_json_path
        self.max_length = max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.add_eos = add_eos
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.loader_workers = loader_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant

    def prepare_data(self):
        if self.cache_dir is None:
            # Just download the dataset
            load_dataset(self.dataset_name, self.dataset_config_name)
        else:
            # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        concat_ids, self.tokenizer = self.process_dataset()
        self.vocab_size = len(self.tokenizer)
        self.dataset_train, self.dataset_val, self.dataset_test = [
            LMDataset(
                concat_ids[split], seq_len=self.max_length, llama2=(self.tokenizer_name == "meta-llama/Llama-2-7b-hf")
            )
            for split in ["train", "validation", "test"]
        ]

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name

        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        if self.raw_json_path is not None:
            raw_datasets = load_dataset("json", data_files=self.raw_json_path)
        else:
            raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)

        # https://github.com/stanford-crfm/mistral/blob/main/src/corpora/auto.py
        if "validation" not in raw_datasets:
            assert "train" in raw_datasets, "You must have train in raw_datasets to make a validation raw_datasets"
            raw_datasets = raw_datasets["train"].train_test_split(
                test_size=self.val_ratio,
                seed=self.val_split_seed,
                shuffle=True,  # Otherwise test will be at the end of the dataset
            )
            raw_datasets["validation"] = raw_datasets["test"]

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        if self.add_eos:
            add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
            add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
            tokenize = lambda example: tokenizer(add_eos_batched(example[text_column_name]))
        else:
            tokenize = lambda example: tokenizer(example[text_column_name])

        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32

        def tokenize_concat(examples):
            # We just need 'input_ids', not 'attention_mask' (since it's all 1)
            input_ids = np.fromiter(chain(*tokenize(examples)["input_ids"]), dtype=dtype)
            # Need to return a list since we're doing batched processing
            return {"input_ids": [input_ids], "len": [len(input_ids)]}

        tokenized_datasets = raw_datasets.map(
            tokenize_concat,
            batched=True,
            num_proc=max(self.num_workers, 1),
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

        # Use disk
        concat_ids = {}
        assert cache_dir is not None
        cache_dir.mkdir(parents=True, exist_ok=True)

        def write_ids_to_disk(example, filename):
            with open(filename, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0)
                start_idx = example["len_offset"] - len(example["input_ids"])
                array_len = len(example["input_ids"])
                arr = np.ndarray((array_len,), dtype=dtype, buffer=mm, offset=np.dtype(dtype).itemsize * start_idx)
                arr[:] = example["input_ids"]
                mm.flush()

        for name, ds in tokenized_datasets.items():
            tokenized_datasets[name] = ds.add_column("len_offset", np.cumsum(ds["len"]))
            array_len = tokenized_datasets[name][-1]["len_offset"]

            filename = cache_dir / f"{name}.bin"

            # Need to create the file with this specific size first
            # https://ostechnix.com/create-files-certain-size-linux/
            subprocess.run(["truncate", "-s", str(array_len * np.dtype(dtype).itemsize), str(filename)], check=True)

            tokenized_datasets[name].map(
                write_ids_to_disk,
                fn_kwargs={"filename": filename},  # .bin
                batched=False,
                num_proc=max(self.num_workers, 1),
                desc="Concatenating examples",
            )
            concat_ids[name] = np.memmap(filename, dtype=dtype, mode="r", shape=(array_len,))

        if cache_dir is not None:
            self._save_to_cache(concat_ids, tokenizer, cache_dir)

            for name in concat_ids:
                Path(cache_dir / f"{name}.bin").unlink()

        return concat_ids, tokenizer

    def _save_to_cache(self, concat_ids, tokenizer, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        master_print(f"Saving to cache at {str(cache_dir)}")
        for k, v in concat_ids.items():
            np.save(cache_dir / f"{k}.npy", v)
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        master_print(f"Load from cache at {str(cache_dir)}")
        concat_ids = {
            split: np.load(cache_dir / f"{split}.npy", mmap_mode="r") for split in ["train", "validation", "test"]
        }
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return concat_ids, tokenizer

    @property
    def _cache_dir_name(self):
        return (
            f"tokenizer_name-{self.tokenizer_name}-val_ratio-{self.val_ratio}-"
            f"val_split_seed-{self.val_split_seed}-add_eos-{self.add_eos}-detokenize-False"
        )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader"""
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            sampler = RandomFaultTolerantSampler(self.dataset_train)
        else:
            shuffle = self.shuffle
            sampler = None

        return self._data_loader(self.dataset_train, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader"""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader"""
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.loader_workers,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
