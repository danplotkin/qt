import os
import torch
from transformers import GPT2Tokenizer, GPT2TokenizerFast

from utils.transformer.model import QT
from utils.configs import load_configs
from utils.tokenizer import get_tokenizer

class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

logo_str = f'''
{bcolors.BOLD}{bcolors.CYAN}           ___    {bcolors.ENDC}{bcolors.BLUE}    ___             {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}          /\  \   {bcolors.ENDC}{bcolors.BLUE}   /\  \            {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}         /88\  \  {bcolors.ENDC}{bcolors.BLUE}   \8\  \           {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}        /8/\8\  \ {bcolors.ENDC}{bcolors.BLUE}    \8\  \          {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}        \8\~\8\__\{bcolors.ENDC}{bcolors.BLUE}    /88\  \         {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}         \8\/8/  /{bcolors.ENDC}{bcolors.BLUE}   /8/\8\__\        {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}          \88/  / {bcolors.ENDC}{bcolors.BLUE}  /8/  \/__/        {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}          /8/__/  {bcolors.ENDC}{bcolors.BLUE} /8/  /             {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}          \8\__\  {bcolors.ENDC}{bcolors.BLUE} \/__/              {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}           \/__/  {bcolors.ENDC}{bcolors.BLUE}                    {bcolors.ENDC}                                   
'''

info_string = '''

A 100M parameter language model created by Daniel Plotkin, Jack Hanke, and Nicole Birova.

Enter 'q' or 'quit' to exit chat window.

'''

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clear terminal header and print info
os.system('clear')
print(logo_str)
print(info_string)

## load model with configs
model_dir = f'./experiments/qt-finetuned/'
# model_dir = f'./experiments/qt-pretrain/'
config = load_configs(path=model_dir+f'config.yaml')
tokenizer = get_tokenizer()
model = QT(
    config=config['transformer'],
    tokenizer=tokenizer,
    device=device,
)
model_dict = torch.load(model_dir+'checkpoints/qt-finetuned_best.pt')
# model_dict = torch.load(model_dir+'checkpoints/qt-pretrain_best.pt')
model.load_state_dict(model_dict['model_state_dict'])
model.eval()


# 
user_prompt_string = f'  Talk to {bcolors.CYAN}q{bcolors.ENDC}{bcolors.BLUE}T{bcolors.ENDC}: '

context = ''
while True:
    user_input = str(input(user_prompt_string))

    if user_input in ['q', 'quit']:
        print('')
        break
    
    # add user input to context
    context += user_input
    # cutoff context
    context = context[:1024]

    model_response = model.decode(
        text = context,
        return_stream = False,
        method = 'topk',
        max_tokens=config['transformer'].max_seq_length
    )

    print(f"{bcolors.CYAN}{model_response:>12}{bcolors.ENDC}")

    # add response to context
    context += model_response



