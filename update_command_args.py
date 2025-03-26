"""
This script is used to add or update arguments in a shell script. For example,

```bash
python update_command_args.py scripts/train/tulu3/grpo_fast_8b.sh \
    --cluster ai2/augusta-google-1 \
    --priority normal \
    --image costah/open_instruct_dev0320_11 | uv run bash
```

would replace the `--cluster`, `--priority`, `--image` arguments in the script with the ones specified.
"""

import sys
import re
import argparse

def read_shell_script(filename):
    with open(filename, 'r') as f:
        return f.read()

def modify_command(content, new_args):
    # Split content into lines while preserving line continuations
    lines = content.replace('\\\n', ' ').split('\n')
    
    # Join all non-empty lines to get the full command
    command = ' '.join(line.strip() for line in lines if line.strip())
    
    # For each new argument
    for arg, value in new_args.items():
        arg_pattern = f"--{arg} [^ ]+"
        if re.search(arg_pattern, command):
            # Replace existing argument
            command = re.sub(arg_pattern, f"--{arg} {value}", command)
        else:
            # Add new argument at the end
            command = f"{command} --{arg} {value}"
    
    # Reformat with line continuations every 2 arguments
    parts = command.split(' --')
    formatted = parts[0]  # First part (python mason.py)
    for i, part in enumerate(parts[1:], 1):
        if i % 2 == 1:
            formatted += f" \\\n    --{part}"
        else:
            formatted += f" --{part}"
    
    return formatted

def main():
    if len(sys.argv) < 2:
        print("Usage: python launch.py <shell_script> [--arg value ...]")
        sys.exit(1)
    
    script_file = sys.argv[1]
    
    # Parse remaining arguments as key-value pairs
    parser = argparse.ArgumentParser()
    for i in range(2, len(sys.argv), 2):
        if i + 1 < len(sys.argv):
            arg = sys.argv[i].lstrip('-')
            parser.add_argument(f"--{arg}")
    
    args = parser.parse_args(sys.argv[2:])
    new_args = {k: v for k, v in vars(args).items() if v is not None}
    
    # Read and modify the script
    content = read_shell_script(script_file)
    modified_content = modify_command(content, new_args)
    
    print(modified_content)

if __name__ == "__main__":
    main() 