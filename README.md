# Introducing... qT!

```
           ___        ___             
          /\  \      /\  \            
         /88\  \     \8\  \           
        /8/\8\  \     \8\  \          
        \8\~\8\__\    /88\  \         
         \8\/8/  /   /8/\8\__\        
          \88/  /   /8/  \/__/        
          /8/__/   /8/  /             
          \8\__\   \/__/              
           \/__/                                                  
```

A 1B parameter language model created by Daniel Plotkin, Jack Hanke, and Nicole Birova.

## Model Card

Tokenizer:
- [GPT2 Tokenizer](https://github.com/huggingface/tokenizers)

TODO

## Data

For pretraining, in order of training...
- [Reddit Comments](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments) (109GB)
- [2022 English Wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia) (20GB)
- [MiniPile](https://huggingface.co/datasets/JeanKaddour/minipile) (5.6GB)
- bookscorpus (4.4GB)

For instruction tuning:
- [lightnovel-2048](https://huggingface.co/datasets/Chat-Error/lightnovel-2048)  (684MB)
- [No_Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (11 MB)

## Project TODOs

- Model design:
    - Change parameter precision to `fp32`
    - ✅ Add smart weight init (word distribution init)
    - ✅ [ALiBi](https://arxiv.org/pdf/2108.12409) instead of sinusoidal
    - ✅ Tie embeddings
    - ✅ Architecture parameters: 
        - `tgt_vocab_size` = `50257`
        - `d_model` = `2048`
        - `d_ff` = `8192`
        - `max_seq_length` = `2048`
        - `num_heads` = `16`
        - `num_layers` = `14`
        - `dropout` = `0.1`

        qT has `1,042,538,496` parameters (tied embeddings):
        - `939k` non-embedding parameters
        - `103k` embedding parameters

        Training with AdamW for `fp32` params with take up atleast `24GBs`, and for inference the model will take up `5GBs`.

- Training:
    - Gradient clipping
    - Perplexity support for training (replace acc metric?)
    - ✅ External logging
- Data:
    - Decide how to best pull, tokenize, and save data
    - Fetch scripts for each source
    - Script to load portion of data into gpu, include offset

## Developer Setup

After first cloning the repo, please set up your env using this command.

```bash
source ./setup_env.sh
```

Before writing any code, make sure your venv is activated, you have pulled changes, and all packages are installed.


When training on a server that you plan to disconnect from, run:

```bash
./train.sh
```

