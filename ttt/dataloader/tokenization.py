import os
from pathlib import Path

current_dir = Path(__file__).parent.absolute()


import pytest

import torch

import dotenv

from ttt.dataloader.language_modeling_hf import LMDataModule

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


# https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python/55423170#55423170
def num_cpu_cores():
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except ImportError:
        return len(os.sched_getaffinity(0))


class TestLMDataModule:
    def test_the_pile(self):
        batch_size = 8
        dataset_name = "the_pile"
        dataset_config_name = None
        cache_dir = Path(
            "/mnt/disks/persistent/the_pile_release"
        )  # TODO: Fill in your path to save the tokenized dataset
        raw_json_path = (
            "/mnt/disks/persistent/PILE"
        )  # TODO: Fill in your path that already stores the raw dataset in json format
        max_length = 2048
        num_workers = num_cpu_cores() // 2
        datamodule = LMDataModule(
            dataset_name,
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            dataset_config_name=dataset_config_name,
            max_length=max_length,
            cache_dir=cache_dir,
            raw_json_path=raw_json_path,
            add_eos=True,  # bos is added by default in llama2 tokenizer
            batch_size=batch_size,
            num_workers=num_workers,
        )
        datamodule.prepare_data()
        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()
        # Token number of The Pile when tokenized by the llama2 tokenizer
        train_len = 383509963636
        val_len = 393983786
        test_len = 383707892
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_books(self):
        batch_size = 8
        dataset_name = "books3"
        dataset_config_name = None
        cache_dir = Path(
            "/mnt/disks/persistent/books3_release"
        )  # TODO: fill in your path to save the tokenized dataset
        raw_json_path = (
            "/mnt/disks/persistent/lwm_raw/lwm_text_data/combined_books.jsonl"
        )  # TODO: fill in your path that already stores the raw dataset in json format
        max_length = 2048
        num_workers = 1
        datamodule = LMDataModule(
            dataset_name,
            tokenizer_name="meta-llama/Llama-2-7b-hf",
            dataset_config_name=dataset_config_name,
            max_length=max_length,
            cache_dir=cache_dir,
            raw_json_path=raw_json_path,
            add_eos=True,  # bos is added by default in llama2 tokenizer
            batch_size=batch_size,
            num_workers=num_workers,
        )
        datamodule.prepare_data()
        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()
        # Token number of Books3 when tokenized by the llama2 tokenizer
        train_len = 32585931901
        val_len = 14007763
        test_len = 14007763
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])
