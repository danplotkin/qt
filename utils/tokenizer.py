from transformers import GPT2TokenizerFast

def get_tokenizer() -> any:
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>"
    })
    return tokenizer
