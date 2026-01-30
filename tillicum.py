"""
Tillicum Job Launcher

A Slurm job submission tool for the UW Tillicum GPU cluster, modeled after mason.py.
Tillicum uses a usage-based model with QOS (Quality of Service) instead of partitions.

Docs: https://hyak.uw.edu/docs/tillicum/scheduling-jobs

Key Tillicum specs:
- 8 NVIDIA H200 GPUs per node (141GB each)
- ~200GB system RAM and 8 CPUs per GPU
- Must request at least 1 GPU (CPU-only jobs not allowed)
- Billing: $0.90/GPU hour

QOS limits:
- normal: 24 hours, 16 GPUs max, 96 concurrent GPUs
- debug: 1 hour, 1 GPU max, 1 job
- interactive: 8 hours, 2 GPUs max, 2 jobs
- long/wide: by request
"""

import argparse
import os
import re
import secrets
import shlex
import string
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.text import Text

console = Console()

# ----------------------------------------------------------------------
# Environment variables to pass through from local environment
WANDB_PASSTHROUGH_VARS = ["WANDB_ENTITY", "WANDB_PROJECT", "WANDB_TAGS"]
HF_PASSTHROUGH_VARS = ["HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HF_TOKEN"]

# ----------------------------------------------------------------------
# Open Instruct logic (from mason.py)
OPEN_INSTRUCT_COMMANDS = [
    "open_instruct/finetune.py",
    "open_instruct/dpo.py",
    "open_instruct/dpo_tune_cache.py",
    "open_instruct/grpo_fast.py",
    "open_instruct/reward_modeling.py",
]

OPEN_INSTRUCT_RESUMABLES = ["open_instruct/grpo_fast.py"]

# ----------------------------------------------------------------------
# Tillicum QOS configuration
QOS_CONFIG = {
    "normal": {"max_time": "24:00:00", "max_gpus": 16, "description": "Standard production work"},
    "debug": {"max_time": "01:00:00", "max_gpus": 1, "description": "Quick testing and setup"},
    "interactive": {"max_time": "08:00:00", "max_gpus": 2, "description": "Real-time work or debugging"},
    "long": {"max_time": None, "max_gpus": None, "description": "Special long jobs (by request)"},
    "wide": {"max_time": None, "max_gpus": None, "description": "Distributed jobs (by request)"},
}

# Default environment variables for training
DEFAULT_ENV_VARS = {
    "RAY_CGRAPH_get_timeout": "300",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "NCCL_DEBUG": "ERROR",
    "VLLM_LOGGING_LEVEL": "WARNING",
    "VLLM_USE_V1": "1",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
}

# Tillicum-specific cache paths (uses $USER which expands at runtime)
# These replace the AI2 /weka paths
TILLICUM_CACHE_VARS = {
    "HF_HOME": "/gpfs/scrubbed/$USER/.cache/huggingface",
    "HF_DATASETS_CACHE": "/gpfs/scrubbed/$USER/.cache/huggingface/datasets",
    "HF_HUB_CACHE": "/gpfs/scrubbed/$USER/.cache/huggingface/hub",
    "TRITON_CACHE_DIR": "/gpfs/scrubbed/$USER/.cache/triton",
}

# Default dataset cache directory for open-instruct (replaces local_dataset_cache)
TILLICUM_DATASET_CACHE_DIR = "/gpfs/scrubbed/$USER/.cache/open_instruct_dataset_cache"


def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


global_wandb_id = generate_id()


def generate_experiment_dir(job_name: str) -> str:
    """Generate a unique experiment directory path under /gpfs/scrubbed/$USER/experiments/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = generate_id(6)
    # Clean job name for filesystem (replace spaces/special chars)
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_name)
    return f"/gpfs/scrubbed/$USER/experiments/{clean_name}_{timestamp}_{unique_id}"


def parse_env_var(env_var_str: str) -> dict[str, str]:
    """Parse environment variable string in the format 'name=value'"""
    if "=" not in env_var_str:
        raise argparse.ArgumentTypeError(f"Environment variable must be in format 'name=value', got: {env_var_str}")
    name, value = env_var_str.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Environment variable name cannot be empty")
    return {"name": name, "value": value}


def parse_time(time_str: str) -> str:
    """Validate and return time string in Slurm format (HH:MM:SS or D-HH:MM:SS)"""
    # Simple validation - Slurm accepts various formats
    return time_str


def get_args():
    parser = argparse.ArgumentParser(
        description="Submit jobs to the UW Tillicum GPU cluster via Slurm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a simple Python script with 1 GPU
  # Commands are automatically wrapped with 'uv run'
  uv run python tillicum.py --gpus 1 --time 01:00:00 -- python my_script.py

  # Run training with 4 GPUs
  uv run python tillicum.py --gpus 4 --time 08:00:00 --job_name my_training -- \\
      python open_instruct/finetune.py --model_name_or_path meta-llama/Llama-3-8B

  # Full node (8 GPUs) training
  uv run python tillicum.py --gpus 8 --time 12:00:00 --job_name fullnode -- \\
      python open_instruct/grpo_fast.py --num_learners_per_node 4 --vllm_num_engines 4

  # Multi-node training with 2 nodes x 8 GPUs each (16 GPUs total)
  # (accelerate args are automatically added for multi-node!)
  uv run python tillicum.py --gpus 8 --nodes 2 --time 12:00:00 --job_name multinode -- \\
      accelerate launch --num_processes 8 \\
      open_instruct/grpo_fast.py --model_name_or_path meta-llama/Llama-3-8B

  # Without uv (if you have your own environment setup)
  uv run python tillicum.py --gpus 1 --time 01:00:00 --no_uv -- python my_script.py

QOS Options:
  normal      24 hours max, up to 16 GPUs per job (default)
  debug       1 hour max, 1 GPU only
  interactive 8 hours max, up to 2 GPUs
  long/wide   By special request
""",
    )
    parser.add_argument(
        "--qos",
        type=str,
        choices=list(QOS_CONFIG.keys()),
        default="normal",
        help="Quality of Service tier (default: normal)",
    )
    parser.add_argument(
        "--gpus", type=int, required=True, help="Number of GPUs per node (required, must be >= 1). Each GPU includes ~200GB RAM and 8 CPUs."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1). Use 2 for multi-node distributed training.",
    )
    parser.add_argument(
        "--time",
        type=parse_time,
        required=True,
        help="Wall time limit in HH:MM:SS or D-HH:MM:SS format",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="tillicum_job",
        help="Name for the Slurm job",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: slurm-%%j.out where %%j is job ID)",
    )
    parser.add_argument(
        "--error",
        type=str,
        default=None,
        help="Error file path (default: same as output)",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=None,
        help="CPUs per task (default: 8 per GPU). Max 8 CPUs per GPU.",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default=None,
        help="Memory allocation (default: 200G per GPU). Max ~200GB per GPU.",
    )
    parser.add_argument(
        "--env",
        type=parse_env_var,
        action="append",
        help="Additional environment variables in format 'name=value'. Can be specified multiple times.",
        default=[],
    )
    parser.add_argument(
        "--module",
        type=str,
        action="append",
        help="Modules to load (e.g., cuda). Can be specified multiple times.",
        default=[],
    )
    parser.add_argument(
        "--no_uv",
        action="store_true",
        help="Don't wrap commands with 'uv run' (uv is used by default)",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default=None,
        help="Working directory for the job (default: current directory)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run as interactive job using salloc instead of sbatch",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the Slurm script without submitting",
    )
    parser.add_argument(
        "--resumable",
        action="store_true",
        help="Enable resumable mode (sets WANDB_RUN_ID and WANDB_RESUME)",
    )
    parser.add_argument(
        "--no_default_env",
        action="store_true",
        help="Don't set default environment variables (VLLM, NCCL, etc.)",
    )
    parser.add_argument(
        "--no_wandb_passthrough",
        action="store_true",
        help="Don't pass through WANDB_ENTITY, WANDB_PROJECT, WANDB_TAGS from local environment",
    )
    parser.add_argument(
        "--no_hf_passthrough",
        action="store_true",
        help="Don't pass through HF_HOME, HF_DATASETS_CACHE, HF_HUB_CACHE, HF_TOKEN from local environment",
    )
    parser.add_argument(
        "--no_accelerate_rewrite",
        action="store_true",
        help="Don't automatically rewrite accelerate launch commands for multi-node",
    )

    # Parse mason args from command args
    tillicum_args, command_args = parser.parse_known_args()
    commands = parse_commands(command_args) if command_args else []

    # Validate GPU count
    if tillicum_args.gpus < 1:
        parser.error("Tillicum requires at least 1 GPU. CPU-only jobs are not allowed.")

    # Validate QOS limits (total GPUs = gpus_per_node * nodes)
    total_gpus = tillicum_args.gpus * tillicum_args.nodes
    qos_config = QOS_CONFIG[tillicum_args.qos]
    if qos_config["max_gpus"] is not None and total_gpus > qos_config["max_gpus"]:
        parser.error(
            f"QOS '{tillicum_args.qos}' allows max {qos_config['max_gpus']} GPUs total, "
            f"but {total_gpus} requested ({tillicum_args.gpus} GPUs x {tillicum_args.nodes} nodes)."
        )

    # Warn about multi-node with non-normal QOS
    if tillicum_args.nodes > 1 and tillicum_args.qos != "normal":
        console.log(f"[yellow]Warning: Multi-node jobs typically require 'normal' QOS, not '{tillicum_args.qos}'[/yellow]")

    # Auto-detect resumable mode for open-instruct commands
    if commands and not tillicum_args.resumable:
        for cmd in commands:
            for target in OPEN_INSTRUCT_RESUMABLES:
                if target in cmd:
                    tillicum_args.resumable = True
                    console.log(f"[yellow]Auto-enabled resumable mode for {target}[/yellow]")
                    break

    return tillicum_args, commands


def parse_commands(command_args: list[str]) -> list[list[str]]:
    """Parse commands separated by '--'.

    Input: ['--', 'which', 'python', '--', 'echo', 'hello']
    Output: [['which', 'python'], ['echo', 'hello']]
    """
    if not command_args:
        return []

    if command_args[0] != "--":
        msg = (
            "Please separate the command you want to run with ' -- ', like "
            "`tillicum [tillicum-args] -- python [python-args]`."
        )
        raise Exception(msg)

    commands = []
    command = []
    for item in command_args:
        if item == "--":
            if command:
                commands.append(command)
                command = []
        else:
            command.append(item)
    if command:
        commands.append(command)
    return commands


def rewrite_accelerate_for_multinode(command: list[str], num_nodes: int, gpus_per_node: int) -> list[str]:
    """Rewrite accelerate launch command for multi-node execution.

    Automatically injects:
    - --num_machines N
    - --machine_rank $NODE_RANK
    - --main_process_ip $MASTER_ADDR
    - --main_process_port 29500

    Also scales --num_processes if specified.
    """
    # Check if this is an accelerate launch command
    if "accelerate" not in command or "launch" not in command:
        return command

    command = command.copy()  # Don't modify original
    total_gpus = num_nodes * gpus_per_node

    # Find the position after "launch"
    try:
        launch_idx = command.index("launch")
    except ValueError:
        return command

    insert_idx = launch_idx + 1

    # Check what's already specified
    cmd_str = " ".join(command)
    has_num_machines = "--num_machines" in cmd_str
    has_machine_rank = "--machine_rank" in cmd_str
    has_main_process_ip = "--main_process_ip" in cmd_str
    has_main_process_port = "--main_process_port" in cmd_str

    # Build args to insert
    args_to_insert = []

    if not has_num_machines:
        args_to_insert.extend(["--num_machines", str(num_nodes)])

    if not has_machine_rank:
        args_to_insert.extend(["--machine_rank", "$NODE_RANK"])

    if not has_main_process_ip:
        args_to_insert.extend(["--main_process_ip", "$MASTER_ADDR"])

    if not has_main_process_port:
        args_to_insert.extend(["--main_process_port", "29500"])

    # Insert the new args after "launch"
    for i, arg in enumerate(args_to_insert):
        command.insert(insert_idx + i, arg)

    # Scale --num_processes if present (multiply by num_nodes)
    # e.g., --num_processes 8 with 2 nodes becomes --num_processes 16
    for i, arg in enumerate(command):
        if arg == "--num_processes" and i + 1 < len(command):
            try:
                current_procs = int(command[i + 1])
                # Only scale if it looks like single-node count
                if current_procs == gpus_per_node:
                    command[i + 1] = str(total_gpus)
                    console.log(f"[yellow]Scaled --num_processes from {current_procs} to {total_gpus} for multi-node[/yellow]")
            except ValueError:
                pass  # Not a number, skip

    return command


def get_passthrough_env_vars() -> dict[str, str]:
    """Get environment variables to pass through from local environment."""
    env_vars = {}

    # WANDB variables
    for var in WANDB_PASSTHROUGH_VARS:
        if var in os.environ:
            env_vars[var] = os.environ[var]

    # HuggingFace variables
    for var in HF_PASSTHROUGH_VARS:
        if var in os.environ:
            env_vars[var] = os.environ[var]

    return env_vars


def build_slurm_script(args: argparse.Namespace, commands: list[list[str]], experiment_dir: str) -> str:
    """Generate a Slurm batch script for Tillicum."""
    # Expand $USER for SBATCH directives (they don't expand shell vars)
    experiment_dir_expanded = experiment_dir.replace("$USER", os.environ.get("USER", "unknown"))
    
    lines = ["#!/bin/bash"]

    # SBATCH directives
    lines.append(f"#SBATCH --job-name={args.job_name}")
    lines.append(f"#SBATCH --qos={args.qos}")
    lines.append(f"#SBATCH --gres=gpu:{args.gpus}")
    lines.append(f"#SBATCH --time={args.time}")

    # Multi-node configuration
    if args.nodes > 1:
        lines.append(f"#SBATCH --nodes={args.nodes}")
        lines.append("#SBATCH --ntasks-per-node=1")  # 1 task per node, distributed training handles GPU parallelism

    # CPUs: default to 8 per GPU
    cpus = args.cpus_per_task if args.cpus_per_task else args.gpus * 8
    lines.append(f"#SBATCH --cpus-per-task={cpus}")

    # Memory: default to 200G per GPU
    mem = args.mem if args.mem else f"{args.gpus * 200}G"
    lines.append(f"#SBATCH --mem={mem}")

    # Output/error files - use experiment directory (expanded, since SBATCH doesn't expand $USER)
    if args.output:
        lines.append(f"#SBATCH --output={args.output}")
    else:
        lines.append(f"#SBATCH --output={experiment_dir_expanded}/logs/slurm-%j.out")

    if args.error:
        lines.append(f"#SBATCH --error={args.error}")
    else:
        lines.append(f"#SBATCH --error={experiment_dir_expanded}/logs/slurm-%j.err")

    lines.append("")

    # Create experiment directory structure
    lines.append("# Create experiment directory structure")
    lines.append(f"export EXPERIMENT_DIR={experiment_dir_expanded}")
    lines.append("mkdir -p $EXPERIMENT_DIR/logs $EXPERIMENT_DIR/output $EXPERIMENT_DIR/checkpoints $EXPERIMENT_DIR/rollouts")
    lines.append('echo "Experiment directory: $EXPERIMENT_DIR"')
    lines.append("")

    # Job info header
    lines.append("# Job information")
    lines.append('echo "=========================================="')
    lines.append('echo "Tillicum Job Started"')
    lines.append('echo "Job ID: $SLURM_JOB_ID"')
    lines.append('echo "Job Name: $SLURM_JOB_NAME"')
    lines.append('echo "Node: $SLURM_NODELIST"')
    lines.append('echo "GPUs: $CUDA_VISIBLE_DEVICES"')
    lines.append('echo "Experiment Dir: $EXPERIMENT_DIR"')
    lines.append('echo "Start Time: $(date)"')
    lines.append('echo "=========================================="')
    lines.append("")

    # Module loading (e.g., cuda)
    if args.module:
        lines.append("# Load modules")
        for module in args.module:
            lines.append(f"module load {module}")
        lines.append("")
        # Set CUDA_HOME if not already set (needed for DeepSpeed)
        lines.append("# Ensure CUDA_HOME is set (required for DeepSpeed)")
        lines.append('if [ -z "$CUDA_HOME" ]; then')
        lines.append('    if [ -d "/usr/local/cuda" ]; then')
        lines.append('        export CUDA_HOME=/usr/local/cuda')
        lines.append('    elif [ -n "$CUDA_PATH" ]; then')
        lines.append('        export CUDA_HOME=$CUDA_PATH')
        lines.append("    fi")
        lines.append("fi")
        lines.append('echo "CUDA_HOME: $CUDA_HOME"')
        lines.append("")

    # Working directory
    if args.working_dir:
        lines.append("# Change to working directory")
        lines.append(f"cd {args.working_dir}")
    else:
        lines.append("# Change to submission directory")
        lines.append(f"cd {os.getcwd()}")
    lines.append("")

    # Environment variables
    lines.append("# Environment variables")

    # Default env vars (unless disabled)
    if not args.no_default_env:
        for name, value in DEFAULT_ENV_VARS.items():
            lines.append(f"export {name}={shlex.quote(value)}")

    # Tillicum-specific cache paths (uses /gpfs/scrubbed instead of /weka)
    lines.append("")
    lines.append("# Tillicum cache paths (uses scratch space)")
    for name, value in TILLICUM_CACHE_VARS.items():
        # Don't quote - we want $USER to expand at runtime
        lines.append(f"export {name}={value}")
    # Dataset cache for open-instruct
    lines.append(f"export DATASET_LOCAL_CACHE_DIR={TILLICUM_DATASET_CACHE_DIR}")
    lines.append("# Create cache directories if they don't exist")
    lines.append("mkdir -p $HF_HOME $HF_DATASETS_CACHE $HF_HUB_CACHE $TRITON_CACHE_DIR $DATASET_LOCAL_CACHE_DIR 2>/dev/null || true")
    lines.append("")

    # Passthrough env vars from local environment (these override Tillicum defaults)
    passthrough_vars = get_passthrough_env_vars()
    wandb_vars_added = []
    hf_vars_added = []

    for name, value in passthrough_vars.items():
        # Check if it's a WANDB or HF variable and if passthrough is enabled
        if name in WANDB_PASSTHROUGH_VARS and not args.no_wandb_passthrough:
            lines.append(f"export {name}={shlex.quote(value)}")
            wandb_vars_added.append(name)
        elif name in HF_PASSTHROUGH_VARS and not args.no_hf_passthrough:
            # HF cache vars from local env override Tillicum defaults
            lines.append(f"export {name}={shlex.quote(value)}")
            hf_vars_added.append(name)

    # User-specified env vars (these override passthrough)
    for env_var in args.env:
        lines.append(f"export {env_var['name']}={shlex.quote(env_var['value'])}")

    # Resumable mode env vars
    if args.resumable:
        lines.append(f"export WANDB_RUN_ID={global_wandb_id}")
        lines.append("export WANDB_RESUME=allow")

    # Multi-node distributed training environment
    if args.nodes > 1:
        lines.append("")
        lines.append("# Multi-node distributed training setup")
        lines.append("# Get head node IP for Ray cluster coordination")
        lines.append("HEAD_NODE_IP=$(hostname -I | awk '{print $1}')")
        lines.append("RAY_PORT=6379")
        lines.append("IP_HEAD=$HEAD_NODE_IP:$RAY_PORT")
        lines.append('echo "Head node IP: $HEAD_NODE_IP"')
        lines.append('echo "Ray head address: $IP_HEAD"')
        lines.append("")
        lines.append("# NCCL settings for multi-node communication")
        lines.append("export NCCL_IB_DISABLE=0")
        lines.append("export NCCL_NET_GDR_LEVEL=2")
        lines.append("export NCCL_CUMEM_ENABLE=0  # Performance fix for vLLM")

    lines.append("")

    # Commands
    lines.append("# Run commands")
    for i, command in enumerate(commands):
        if len(commands) > 1:
            lines.append(f'echo "--- Running command {i + 1} ---"')

        # Apply accelerate rewrite for multi-node if enabled
        if args.nodes > 1 and not args.no_accelerate_rewrite:
            command = rewrite_accelerate_for_multinode(command, args.nodes, args.gpus)

        # Wrap with uv run if enabled (default)
        if not args.no_uv:
            command = ["uv", "run"] + command

        # Quote arguments properly, using double quotes for shell variables (allows expansion)
        quoted_parts = []
        for arg in command:
            if "$" in arg:
                # Use double quotes for arguments with shell variables (allows expansion)
                # Escape any existing double quotes and backticks in the argument
                escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
                quoted_parts.append(f'"{escaped}"')
            else:
                quoted_parts.append(shlex.quote(arg))
        cmd_str = " ".join(quoted_parts)

        # For multi-node jobs, use ray symmetric-run (Ray 2.49+)
        # This starts Ray on all nodes and runs the command ONLY on the head node
        if args.nodes > 1:
            lines.append(f"# Using ray symmetric-run for {args.nodes}-node cluster")
            lines.append(f"srun --nodes={args.nodes} --ntasks={args.nodes} \\")
            lines.append(f"    ray symmetric-run \\")
            lines.append(f"    --address $IP_HEAD \\")
            lines.append(f"    --min-nodes {args.nodes} \\")
            lines.append(f"    --num-gpus {args.gpus} \\")
            lines.append(f"    -- \\")
            lines.append(f"    {cmd_str}")
        else:
            lines.append(cmd_str)
        lines.append("")

    # Job completion footer
    lines.append('echo "=========================================="')
    lines.append('echo "Job Completed: $(date)"')
    lines.append('echo "=========================================="')

    return "\n".join(lines)


def submit_batch_job(script_content: str, experiment_dir: str, dry_run: bool = False) -> str | None:
    """Submit a batch job to Slurm."""
    if dry_run:
        console.rule("[bold blue]Slurm Script (Dry Run)[/bold blue]")
        console.print(Text(script_content))
        return None

    # Create experiment directory and save script there
    # Replace $USER with actual username for the local path
    local_experiment_dir = experiment_dir.replace("$USER", os.environ.get("USER", "unknown"))
    os.makedirs(local_experiment_dir, exist_ok=True)
    
    script_path = os.path.join(local_experiment_dir, "job.slurm")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    console.log(f"Saved script to: [bold]{script_path}[/bold]")

    try:
        result = subprocess.run(
            ["sbatch", script_path], capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
        console.log(f"[green]✓ {output}[/green]")

        # Extract job ID
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            return job_id
        return None
    except subprocess.CalledProcessError as e:
        console.log(f"[red]Error submitting job:[/red]")
        console.log(e.stderr)
        raise


def run_interactive_job(args: argparse.Namespace) -> None:
    """Run an interactive job using salloc."""
    cmd = [
        "salloc",
        f"--qos={args.qos}",
        f"--gpus={args.gpus}",
        f"--cpus-per-task={args.cpus_per_task if args.cpus_per_task else args.gpus * 8}",
        f"--mem={args.mem if args.mem else f'{args.gpus * 200}G'}",
        f"--time={args.time}",
    ]

    # Add multi-node options
    if args.nodes > 1:
        cmd.extend([f"--nodes={args.nodes}", "--ntasks-per-node=1"])

    console.log(f"[bold]Starting interactive session:[/bold]")
    console.log(" ".join(cmd))
    console.log("")
    if args.nodes > 1:
        console.log(f"[yellow]Note: You will be allocated {args.nodes} compute nodes.[/yellow]")
        console.log("[yellow]Use 'srun' to run commands across all nodes.[/yellow]")
    else:
        console.log("[yellow]Note: You will be connected to a compute node.[/yellow]")
    console.log("[yellow]Type 'exit' to end the session.[/yellow]")
    console.log("")

    # Run salloc interactively
    os.execvp("salloc", cmd)


def estimate_cost(gpus_per_node: int, nodes: int, time_str: str) -> float:
    """Estimate job cost based on GPU hours."""
    # Parse time string (HH:MM:SS or D-HH:MM:SS)
    parts = time_str.split("-")
    if len(parts) == 2:
        days = int(parts[0])
        time_part = parts[1]
    else:
        days = 0
        time_part = parts[0]

    h, m, s = map(int, time_part.split(":"))
    total_hours = days * 24 + h + m / 60 + s / 3600

    # $0.90 per GPU hour, total GPUs = gpus_per_node * nodes
    total_gpus = gpus_per_node * nodes
    return total_gpus * total_hours * 0.90


def main():
    args, commands = get_args()

    # Calculate costs upfront
    estimated_cost = estimate_cost(args.gpus, args.nodes, args.time)
    total_gpus = args.gpus * args.nodes

    # Generate experiment directory
    experiment_dir = generate_experiment_dir(args.job_name)

    # Show job info
    if args.dry_run:
        console.rule("[bold yellow]Tillicum Job - DRY RUN[/bold yellow]")
    else:
        console.rule("[bold blue]Tillicum Job Submission[/bold blue]")

    console.log(f"QOS: {args.qos} ({QOS_CONFIG[args.qos]['description']})")
    if args.nodes > 1:
        console.log(f"Nodes: {args.nodes}")
        console.log(f"GPUs per node: {args.gpus}")
        console.log(f"Total GPUs: {total_gpus}")
    else:
        console.log(f"GPUs: {args.gpus}")
    console.log(f"Time: {args.time}")
    console.log(f"Job Name: {args.job_name}")

    # Show experiment directory
    console.log("")
    console.rule("[bold magenta]Experiment Directory[/bold magenta]")
    # Replace $USER with actual user for display
    display_dir = experiment_dir.replace("$USER", os.environ.get("USER", "$USER"))
    console.log(f"[bold]{display_dir}[/bold]")
    console.log(f"  logs/        - Slurm output logs")
    console.log(f"  output/      - Model output (use --output_dir $EXPERIMENT_DIR/output)")
    console.log(f"  checkpoints/ - Training checkpoints")
    console.log(f"  rollouts/    - Rollout traces (use --rollouts_save_path $EXPERIMENT_DIR/rollouts)")

    # Show passthrough env vars
    passthrough_vars = get_passthrough_env_vars()
    wandb_vars = [v for v in passthrough_vars if v in WANDB_PASSTHROUGH_VARS and not args.no_wandb_passthrough]
    hf_vars = [v for v in passthrough_vars if v in HF_PASSTHROUGH_VARS and not args.no_hf_passthrough]

    if wandb_vars or hf_vars:
        console.log("")
        console.log("[dim]Passing through from local environment:[/dim]")
        if wandb_vars:
            console.log(f"  WANDB: {', '.join(wandb_vars)}")
        if hf_vars:
            console.log(f"  HF: {', '.join(hf_vars)}")

    console.log("")

    # Cost summary
    console.rule("[bold green]Cost Estimate[/bold green]")
    console.log(f"GPU Hours: {total_gpus} GPUs × {args.time} = {total_gpus * _parse_hours(args.time):.2f} GPU-hours")
    console.log(f"Rate: $0.90 per GPU-hour")
    console.log(f"[bold]Estimated Cost: ${estimated_cost:.2f}[/bold]")
    console.log("")

    if args.interactive:
        if commands:
            console.log("[yellow]Warning: Commands are ignored in interactive mode[/yellow]")
        if args.dry_run:
            console.log("[yellow]Would run: salloc with the above configuration[/yellow]")
            return
        run_interactive_job(args)
        return

    if not commands:
        if args.dry_run:
            console.log("[dim]No commands specified - showing cost estimate only[/dim]")
            return
        console.log("[red]Error: No commands specified. Use ' -- ' to separate your command.[/red]")
        console.log("[dim]Example: python tillicum.py --gpus 1 --time 01:00:00 -- python my_script.py[/dim]")
        sys.exit(1)

    # Show commands
    for i, command in enumerate(commands):
        console.rule(f"[bold]Command {i + 1}[/bold]")
        console.print(" ".join(command))

    console.log("")

    # Generate script
    script = build_slurm_script(args, commands, experiment_dir)

    if args.dry_run:
        console.rule("[bold cyan]Generated Slurm Script[/bold cyan]")
        console.print(Text(script))
        console.log("")
        console.rule("[bold yellow]DRY RUN - No job submitted[/bold yellow]")
    else:
        console.log("Submitting job to Slurm...")
        job_id = submit_batch_job(script, experiment_dir, dry_run=False)
        if job_id:
            console.log("")
            console.log(f"Monitor job:    [bold]squeue -j {job_id}[/bold]")
            console.log(f"Cancel job:     [bold]scancel {job_id}[/bold]")
            console.log(f"View logs:      [bold]tail -f {display_dir}/logs/slurm-{job_id}.out[/bold]")
            console.log(f"Job efficiency: [bold]seff {job_id}[/bold] (after completion)")


def _parse_hours(time_str: str) -> float:
    """Parse time string to hours for cost calculation."""
    parts = time_str.split("-")
    if len(parts) == 2:
        days = int(parts[0])
        time_part = parts[1]
    else:
        days = 0
        time_part = parts[0]

    h, m, s = map(int, time_part.split(":"))
    return days * 24 + h + m / 60 + s / 3600


if __name__ == "__main__":
    main()
