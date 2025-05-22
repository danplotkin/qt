import os

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
Enter 'q' or 'quit' to exit chat window.
**TODO add model card!

'''

os.system('clear')
print(logo_str)
print(info_string)

user_prompt_string = f'  Talk to {bcolors.CYAN}q{bcolors.ENDC}{bcolors.BLUE}t{bcolors.ENDC}: '

while True:
    user_input = str(input(user_prompt_string))

    if user_input in ['q', 'quit']:
        print('')
        break

    model_response = '''
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    '''

    print(f"{bcolors.CYAN}{model_response:>12}{bcolors.ENDC}")

