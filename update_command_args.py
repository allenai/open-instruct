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

import argparse
import sys


def read_shell_script(filename):
    with open(filename, 'r') as f:
        return f.read()

def modify_command(content, new_args):
    split_content = content.split(" ")
    new_content = []
    flag_args = []
    flag = None
    for _, part in enumerate(split_content):
        if flag is None:
            if not part.startswith('--'):
                new_content.append(part)
            else:
                flag = part.split('--')[1]
                flag_args.append(part)
        else:
            if not part.startswith('--'):
                flag_args.append(part)
            else:
                if flag in new_args:
                    new_content.append(f"--{flag}")
                    new_args_values = new_args[flag]
                    for i in range(len(new_args_values)):
                        if "</" in new_args_values[i]:
                            new_args_values[i] = f"'{new_args_values[i]}'"
                    if isinstance(new_args_values, list):
                        new_content.extend(new_args_values)
                    else:
                        new_content.append(new_args_values)
                        # hack the convention to make the format nicer
                    new_content.extend(["\\\n", "", "", ""])
                    del new_args[flag]
                else:
                    new_content.append(f"--{flag}")
                    if isinstance(flag_args, list):
                        new_content.extend(flag_args)
                    else:
                        new_content.append(flag_args)
                flag = part.split('--')[1]
                flag_args = []
    if flag is not None:
        new_content.append(f"--{flag}")
        if isinstance(flag_args, list):
            new_content.extend(flag_args)
        else:
            new_content.append(flag_args)


    # add the remaining args
    for flag, value in new_args.items():
        new_content.append(f"--{flag}")
        if isinstance(value, list):
            new_content.extend(value)
        else:
            formatted += f" --{part}"

    return formatted

def main():
    if len(sys.argv) < 2:
        print("Usage: python update_command_args.py <shell_script> [--arg value ...]")
        sys.exit(1)

    script_file = sys.argv[1]

    # Parse remaining arguments as key-value pairs
    # NOTE: we need to handle `nargs` for cases like `--dataset_mixer_list xxx 1.0`
    parser = argparse.ArgumentParser()
    num_values = 0
    last_arg = None
    for i in range(2, len(sys.argv)):
        if sys.argv[i].startswith('--'):
            arg = sys.argv[i].lstrip('-')
            nargs = "+" if num_values % 2 == 0 else "?"
            if last_arg is not None:
                parser.add_argument(f"--{last_arg}", nargs=nargs)
            last_arg = arg
            num_values = 0
        else:
            num_values += 1
    nargs = "+" if num_values % 2 == 0 else "?"
    if last_arg is not None:
        parser.add_argument(f"--{last_arg}", nargs=nargs)

    args = parser.parse_args(sys.argv[2:])
    new_args = {k: v for k, v in vars(args).items() if v is not None}
    # Read and modify the script
    content = read_shell_script(script_file)
    modified_content = modify_command(content, new_args)

    print(modified_content)

if __name__ == "__main__":
    main()
