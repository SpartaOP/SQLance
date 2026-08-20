# SQLance: Text-to-SQL Generation Pipeline

## Overview
SQLance is a machine learning pipeline that fine-tunes a Large Language Model to translate natural language questions into executable SQL queries. This project demonstrates the implementation of Parameter-Efficient Fine-Tuning (PEFT) using QLoRA to train the model efficiently. 

Through this pipeline, the model's accuracy in generating valid, context-aware SQL queries improved from a baseline of 0.00% to 94.00% on the evaluation holdout set[cite: 1].

## Repository Contents
* `Text-to-SQL.ipynb`: The core Jupyter Notebook containing the full training, quantization, and evaluation pipeline[cite: 1].

## External Resources
To keep the repository lightweight, massive model weights and datasets are not stored locally. The notebook dynamically fetches them from Hugging Face:
* **Base Model:** [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)[cite: 1]
* **Dataset:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (using a 3,000-sample shuffled subset for training)[cite: 1]

## Prerequisites
To reproduce this environment, you will need Python 3 and the following dependencies[cite: 1]:
* `transformers`
* `peft`
* `bitsandbytes`
* `trl`
* `accelerate`
* `datasets`

*Note: The evaluation script also utilizes Python's built-in `sqlite3` and `re` modules to execute and verify the generated SQL against a memory database[cite: 1].*

## Hardware Requirements
This notebook is optimized for execution on a GPU. The original pipeline was tested and executed using an NVIDIA Tesla T4 GPU[cite: 1]. 

## Usage
1. Open `Text-to-SQL.ipynb` in a Jupyter environment (Google Colab is recommended for easy access to GPUs).
2. Authenticate with Hugging Face when prompted to download the base model[cite: 1].
3. Execute the cells sequentially to install dependencies, load the quantized model, process the dataset, and initialize the `SFTTrainer`[cite: 1]. 
4. The final cells contain the evaluation loop that tests the fine-tuned adapter against the base model[cite: 1].
