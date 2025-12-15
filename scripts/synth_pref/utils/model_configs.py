from typing import Any

default_generation_params = {"temperature": 1, "top_p": 1, "max_tokens": 7500}


MODELS: dict[str, dict[str, Any]] = {
    "Meta-Llama/Llama-3-8B-Instruct": {
        "model": {"name_or_path": "Meta-Llama/Llama-3-8B-Instruct", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
    "Meta-Llama/Llama-3.1-8B-Instruct": {
        "model": {"name_or_path": "Meta-Llama/Llama-3.1-8B-Instruct", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
    "Meta-Llama/Llama-3.1-70B-Instruct": {
        "model": {
            "name_or_path": "Meta-Llama/Llama-3.1-70B-Instruct",
            "tensor_parallel_size": 4,
            "num_scheduler_steps": 1,
        },
        "generate": default_generation_params,
        "task": {"gpu_count": 4},
    },
    "Meta-Llama/Llama-3-405B-Instruct-fp8": {
        "model": {
            "name_or_path": "Meta-Llama/Llama-3-405B-Instruct-fp8",
            "tensor_parallel_size": 8,
            "num_scheduler_steps": 1,
            "max_model_len": 64000,
        },
        "generate": default_generation_params,
        "task": {"gpu_count": 8},
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "model": {"name_or_path": "mistralai/Mistral-7B-Instruct-v0.2", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "model": {
            "name_or_path": "mistralai/Mistral-Nemo-Instruct-2407",
            "tensor_parallel_size": 2,
            "num_scheduler_steps": 1,
        },
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "allenai/tulu-2-13b": {
        "model": {"name_or_path": "allenai/tulu-2-13b", "tensor_parallel_size": 2, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "allenai/tulu-2-70b": {
        "model": {"name_or_path": "allenai/tulu-2-70b", "tensor_parallel_size": 4, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 4},
    },
    "01-ai/Yi-34B-Chat": {
        "model": {"name_or_path": "01-ai/Yi-34B-Chat", "tensor_parallel_size": 2, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "01-ai/Yi-6B-Chat": {
        "model": {"name_or_path": "01-ai/Yi-6B-Chat", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
    "allenai/tulu-2-7b": {
        "model": {"name_or_path": "allenai/tulu-2-7b", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
    "mosaicml/mpt-30b-chat": {
        "model": {"name_or_path": "mosaicml/mpt-30b-chat", "tensor_parallel_size": 2, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "mosaicml/mpt-7b-8k-chat": {
        "model": {"name_or_path": "mosaicml/mpt-7b-8k-chat", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
    "google/gemma-2-27b-it": {
        "model": {"name_or_path": "google/gemma-2-27b-it", "tensor_parallel_size": 2, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "google/gemma-2-9b-it": {
        "model": {"name_or_path": "google/gemma-2-9b-it", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
    "internlm/internlm2_5-20b-chat": {
        "model": {
            "name_or_path": "internlm/internlm2_5-20b-chat",
            "trust_remote_code": True,
            "tensor_parallel_size": 2,
            "num_scheduler_steps": 1,
        },
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "internlm/internlm2_5-7b-chat": {
        "model": {
            "name_or_path": "internlm/internlm2_5-7b-chat",
            "trust_remote_code": True,
            "tensor_parallel_size": 2,
            "num_scheduler_steps": 1,
        },
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "internlm/internlm2_5-1_8b-chat": {
        "model": {
            "name_or_path": "internlm/internlm2_5-1_8b-chat",
            "trust_remote_code": True,
            "tensor_parallel_size": 2,
            "num_scheduler_steps": 1,
        },
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "tiiuae/falcon-7b-instruct": {
        "model": {"name_or_path": "tiiuae/falcon-7b-instruct", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "model": {"name_or_path": "Qwen/Qwen2.5-72B-Instruct", "tensor_parallel_size": 4, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 4},
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "model": {"name_or_path": "Qwen/Qwen2.5-32B-Instruct", "tensor_parallel_size": 2, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "model": {"name_or_path": "Qwen/Qwen2.5-14B-Instruct", "tensor_parallel_size": 2, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "model": {"name_or_path": "Qwen/Qwen2.5-7B-Instruct", "tensor_parallel_size": 2, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "model": {"name_or_path": "Qwen/Qwen2.5-3B-Instruct", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
    "allenai/llama-tulu-3-70b": {
        "model": {"name_or_path": "allenai/llama-tulu-3-70b", "tensor_parallel_size": 4, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 4},
    },
    "allenai/llama-tulu-3-8b": {
        "model": {"name_or_path": "allenai/llama-tulu-3-8b", "tensor_parallel_size": 2, "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 2},
    },
    "allenai/OLMo-7B-0724-SFT-hf": {
        "model": {"name_or_path": "allenai/OLMo-7B-0724-SFT-hf", "num_scheduler_steps": 1},
        "generate": default_generation_params,
        "task": {"gpu_count": 1},
    },
}
