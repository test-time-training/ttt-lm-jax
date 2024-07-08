<h1>Dataset Preparation</h1>

<h2>Option 1: Download Pre-Tokenized Datasets (Recommended)</h2>

Our Llama-2 tokenized datasets are available for download from Google Cloud Buckets:

```
gsutil -m cp -r gs://llama-2-pile/* llama-2-pile/
gsutil -m cp -r gs://llama-2-books3/* llama-2-books3/
```

Once downloaded, set the `dataset_path` flag in `train.py` to the directory containing the `tokenizer_name-meta-llama` folder. This will allow the dataloader to find the correct path.

<h2>Option 2: Tokenize Datasets Yourself</h2>

Since the raw Pile and Books3 datasets are no longer publically available on Huggingface, we recommend acquiring them via correspondence to their authors or from the community.

Before tokenization, set `raw_json_path` and `cache_dir` in `tokenization.py` to the path where the raw dataset (in json format) is stored and where you want to store the tokenized dataset, respectively. 

Our tokenization script is based on [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main/training#dataset-preparation). Tokenize the raw datasets using the commands below.

**Pile:**
```
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s ttt/dataloader/tokenization.py -k "pile"
```
This takes around 20h on a 64-core CPU. The processed dataset is 716G.

**Books3:**
```
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s ttt/dataloader/tokenization.py -k "books"
```
This takes around 3h on a 64-core CPU. The processed dataset is 61G.
