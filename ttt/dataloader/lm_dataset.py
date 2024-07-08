# Inspired by https://github.com/NVIDIA/Megatron-LM/blob/main/tasks/zeroshot_gpt/datasets.py
# Except we don't pad the last block and don't use overlapping eval
# And we return both the input and the target
import math
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import RandomSampler


class RandomFaultTolerantSampler(RandomSampler):
    def __init__(self, *args, generator=None, **kwargs):
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0  # Absolute position of data reading
        self.is_rollback = False
        self.state = self.generator.get_state()  # Record the initial state of generator determined by seed
        # Should not be changed before an entire loop over dataset is done
        # Give same seed, generator state change deterministically after each torch.randperm
        self.shuffle_log = [{"shuffle_after": self.counter}]

    def state_dict(self):
        return {"random_state": self.state, "counter": self.counter, "shuffle_log": self.shuffle_log}

    def load_state_dict(self, state_dict):
        self.state = state_dict["random_state"]
        self.counter = state_dict["counter"]
        if "shuffle_log" in state_dict:
            self.shuffle_log = state_dict["shuffle_log"]  # A list of shuffle records, each record is a dict

    def update_shuffle_history(self):
        self.shuffle_log.append({"shuffle_after": self.counter})

    def go_through_shuffle_history(self):
        N = len(self.data_source)
        initial_shuffle_after = self.shuffle_log[0]["shuffle_after"]
        self.generator.set_state(self.state)
        indices = torch.randperm(N, generator=self.generator)

        for shuffle_record in self.shuffle_log[1:]:
            shuffle_after = shuffle_record["shuffle_after"]
            new_order = torch.randperm(N - shuffle_after, generator=self.generator)  #
            indices = torch.concatenate([indices[:shuffle_after], indices[shuffle_after:][new_order]])

        return indices

    def __iter__(self) -> Iterator[int]:

        if self.is_rollback:
            # Before entering __iter__() due to rollback, set loader.sampler.is_rollback = True manually outside
            self.update_shuffle_history()  # Add a shuffle action at self.counter, which is where we resume from but need a different coming data order
            self.is_rollback = False

        indices = self.go_through_shuffle_history()
        indices = indices[self.counter :].tolist()

        for index in indices:
            self.counter += 1
            yield index

        # End of one loop over the entire dataset
        self.counter = 0
        self.state = self.generator.get_state()  # If have the next epoch, state will definitely be different
        self.shuffle_log = [{"shuffle_after": self.counter}]


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len, drop_last=True, llama2=False):
        """tokens should be a numpy array"""
        self.seq_len = seq_len
        ntokens = len(tokens)
        if drop_last:
            ntokens = ((ntokens - 1) // seq_len) * seq_len + 1
        self.ntokens = ntokens
        # We're careful not to slice tokens, since it could be a memmap'ed array or H5 dataset,
        # and slicing would load it to memory.
        self.tokens = tokens
        self.total_sequences = math.ceil((self.ntokens - 1) / self.seq_len)
        self.llama2 = llama2

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        idx = idx % self.ntokens
        start_idx = idx * self.seq_len
        seq_len = min(self.seq_len, self.ntokens - 1 - start_idx)
        data = torch.as_tensor(self.tokens[start_idx : (start_idx + seq_len + 1)].astype(np.int32))
        if self.llama2:
            return {
                "input_tokens": data[:-1],
                "target_tokens": data[1:].clone(),
                "loss_masks": (data[1:] != 1).to(torch.float32),
            }
        else:
            return {
                "input_tokens": data[:-1],
                "target_tokens": data[1:].clone(),
                "loss_masks": torch.ones_like(data[:-1], dtype=torch.float32),
            }
