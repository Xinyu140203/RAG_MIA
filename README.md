# DCMI: A Differential Calibration Membership Inference Attack Against Retrieval-Augmented Generation



This is the official implementation of the paper "DCMI: A Differential Calibration Membership Inference Attack Against Retrieval-Augmented Generation". The proposed Membership Inference Attack based on Differential-calibration against Retrieval-Augmented Generation (RAG) is implemented as follows.


![image](https://github.com/user-attachments/assets/58cfed47-bb8b-4a04-8db4-a6a923773f83)

## Quick Start

### RAG Construction

#### Corpus Construction
To build an index, you first need to save your corpus as a `jsonl` file with each line representing a document.

```jsonl
{"id": "0", "contents": "..."}
{"id": "1", "contents": "..."}
```

If you want to use Wikipedia as your corpus, you can refer to our documentation [Processing Wikipedia](./docs/original_docs/process-wiki.md) to convert it into an indexable format.

#### Index Construction

You can use the following code to build your own index.

* For **dense retrieval methods**, especially popular embedding models, we use `faiss` to build the index.

* For **sparse retrieval methods (BM25)**, we use `Pyserini` or `bm25s` to build the corpus into a Lucene inverted index. The built index contains the original documents.

##### For Dense Retrieval Methods

Modify the parameters in the following code to your own.

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path /model/e5-base-v2/ \
  --corpus_path indexes/sample_corpus.jsonl \
  --save_dir indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --faiss_type Flat 
```

* ```--pooling_method```: If this parameter is not specified, we will automatically select it based on the model name and model file. However, since different embedding models use different pooling methods, **we may not have fully implemented them**. To ensure accuracy, you can **specify the pooling method corresponding to the retrieval model you are using** (`mean`, `pooler`, or `cls`).

* ```---instruction```: Some embedding models require additional instructions to be concatenated to the query before encoding, which can be specified here. Currently, we will automatically fill in the instructions for **E5** and **BGE** models, while other models need to be supplemented manually.

If the retrieval model supports the `sentence transformers` library, you can use the following code to build the index (**without considering the pooling method**).

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path /model/e5-base-v2/ \
  --corpus_path indexes/sample_corpus.jsonl \
  --save_dir indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --sentence_transformer \
  --faiss_type Flat 
```

##### For Sparse Retrieval Methods (BM25)

If building a bm25 index, there is no need to specify `model_path`.


```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path indexes/sample_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir indexes/ 
```

### RAG Run Methods

Run the experiment on the NQ dataset using the following command.

```bash
python run_exp.py --method_name 'Standard-RAG' \
                  --split 'test' \
                  --dataset_name 'nq' \
                  --gpu_id '0,1,2,3'
```

The method can be selected from the following:
```
Standard-RAG llmlingua SC-RAG ircot spring
```

### Perturbed samples Construction
```bash
python perturb.py 
```


### MIA Based on Differential Calibration
```bash
python MIA.py 
```
