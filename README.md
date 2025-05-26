# qt - A 1B Parameter Language Model

Members: Daniel Plotkin, Jack Hanke, Nicole Birova

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

## Model Card

Tokenizer:
- [Tokenizer](https://github.com/huggingface/tokenizers)

TODO

## Data

For pretraining, in order of training...
- [Reddit Comments](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments) (109GB total)
- [2022 English Wikipedia](https://huggingface.co/datasets/legacy-datasets/wikipedia) (20GB)
- [MiniPile](https://huggingface.co/datasets/JeanKaddour/minipile) (5.6GB)
- bookscorpus (4.4GB)

For instruction tuning:
- [lightnovel-2048](https://huggingface.co/datasets/Chat-Error/lightnovel-2048)  (684MB)
- [No_Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (11 MB)

## Project TODOs

- Model design:
    - Change parameter precision
    - [ALiBi](https://arxiv.org/pdf/2108.12409) instead of sinusoidal
    - Decide architecture parameters: 
        - `tgt_vocab_size` = `50257`
        - `d_model` = `2048`
        - `d_ff` = `8192`
        - `num_heads` = `16`
        - `num_layers` = `14`
        - `max_seq_length` = `512`
        - `dropout` = `0.1`

        This corresponds with `1,042,538,496` parameters, and with tied embeddings:
        - `939k` non-embedding parameters
        - `103k` embedding parameters

        Training with Adam for fp32 params 

    - Tie embeddings
- Training:
    - Gradient clipping
    - External logging
    - 
- Data:
    - Decide how to best pull, tokenize, and save data
    - Fetch scripts for each source
    - Script to load portion of data into gpu, include offset
    - 

## Developer Setup

After first cloning the repo, please set up your env using this command.

```bash
source ./setup_env.sh
```

Before writing any code, make sure your venv is activated, you have pulled changes, and all packages are installed.
