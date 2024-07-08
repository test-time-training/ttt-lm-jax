# Model Documentation

This codebase is implemented in [JAX](https://jax.readthedocs.io/en/latest/index.html) and is based on [EasyLM](https://github.com/young-geng/EasyLM/tree/main).

## Training Flags
- `mesh_dim` refers to the the mesh used by JAX to parallelize computation across multiple accelerators and hosts. Please refer to the [EasyLM paralellization documentation](https://github.com/young-geng/EasyLM/blob/main/docs/parallelism.md) for configuration.
- `seq_length` and `global_batch_size` determine the total number of tokens per batch (fixed to 0.5 million in our paper).
- `load_model_config` is used to load a default configs from `model.py`
- `update_model_config` is used to update a default config. To update specific keys, pass a dictionary to the flag:

```
--update_model_config="dict(seq_modeling_block='ttt_linear', ttt_base_lr=1.0)"
```

All additional hyperparameters are specified Appendix C of our paper.

## Model Flags
All model configuration flags can be found in `model.py`. Here are a few important details to note:

We implement four TTT choices for the `seq_modeling_block`:
  - `ttt_linear` and `ttt_mlp`, which specify TTT layers within the **Mamba backbone**.
  - `ttt_linear_base` and `ttt_mlp_base`, which specify TTT layers within the **Transformer backbone**.

### TTT LR
- For all `ttt_linear` experiments, `ttt_base_lr` is set to 1.0. 
- For all `ttt_mlp` experiments:
  - `ttt_base_lr` is set to 0.1
  - `ttt_base_lr_init` is set to 0.01
  - `ttt_base_lr_warmup` is set to the total number of outer loop warmup steps. 
