# Learning to (Learn at Test Time): RNNs with Expressive Hidden States
[**Paper**](https://arxiv.org/abs/2407.04620)
| [**PyTorch Codebase**](https://github.com/test-time-training/ttt-lm-pytorch)
| [**Setup**](#setup)
| [**Replicating Experiments**](#replicating-experiments)
| [**Model Docs**](ttt/README.md)
| [**Dataset Preparation**](ttt/dataloader/README.md)
| [**Inference Benchmark**](https://github.com/test-time-training/ttt-lm-kernels)

## Abstract

Self-attention performs well in long context but has quadratic complexity. Existing RNN layers
have linear complexity, but their performance in long context is limited by the expressive power
of their hidden state. We propose a new class of sequence modeling layers with linear complexity
and an expressive hidden state. The key idea is to make the hidden state a machine learning
model itself, and the update rule a step of self-supervised learning. 

Since the hidden state is updated by training even on test sequences, our layers are called **Test-Time Training (TTT) layers**.
We consider two instantiations: TTT-Linear and TTT-MLP, whose hidden state is a linear model
and a two-layer MLP respectively. 

## Setup
This codebase is implemented in [JAX](https://jax.readthedocs.io/en/latest/index.html) and has been tested on both GPUs and Cloud TPU VMs with Python 3.11. 

For a PyTorch model definition, please refer to [this link](https://github.com/test-time-training/ttt-lm-pytorch). For inference kernels, or to replicate speed benchmarks from our paper, please view our [kernel implementations](https://github.com/test-time-training/ttt-lm-kernels).

### Environment Installation
To setup and run our code on a (local) GPU machine, we highly recommend using [Anaconda](https://anaconda.com/download) when installing python dependencies. Install GPU requirements using:
```
cd requirements
pip install -r gpu_requirements.txt
```

For TPU, please refer to [this link](https://cloud.google.com/tpu/docs/quick-starts) for guidance on creating cloud TPU VMs. Then, run:
```
cd requirements
pip install -r tpu_requirements.txt
```

### WandB Login
We use WandB for logging training metrics and TTT statistics. After installing the requirements, login to WandB using:
```
wandb login
```


### Dataset Download
Our Llama-2 tokenized datasets are available for download from Google Cloud Buckets:

```
gsutil -m cp -r gs://llama-2-pile/* llama-2-pile/
gsutil -m cp -r gs://llama-2-books3/* llama-2-books3/
```

Once downloaded, set the `dataset_path` flag in `train.py` to the directory containing the `tokenizer_name-meta-llama` folder. This will allow the dataloader to find the correct path. 

Alternatively, to tokenize datasets yourself, refer to [dataset preparation](ttt/dataloader/README.md).

## Replicating Experiments
We provide scripts corresponding to each experiment in our paper in the `scripts` folder. After specifying the experiment name and directory, select the desired context length and divide by 0.5 million to calculate the appropriate batch size. 

Depending on the model size, you may need to modify the `mesh_dim` to introduce model sharding. See the [model docs](ttt/README.md) for additional information on the training configuration.

## Credits
* This codebase is based on [EasyLM](https://github.com/young-geng/EasyLM).
* Our dataloader is based on [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main/training).
