import dataclasses
from typing import Protocol

LEARNER_PLACEMENT_GROUP_CPU_PER_GPU = 10
DATA_PREPARATION_ACTOR_NUM_CPUS = 2


class ExperimentConfigLike(Protocol):
    num_learners_per_node: list[int]
    single_gpu_mode: bool


class VLLMConfigLike(Protocol):
    vllm_num_engines: int
    vllm_tensor_parallel_size: int


@dataclasses.dataclass(frozen=True)
class GrpoFastResourcePlan:
    """Minimum resource view needed to reason about GRPO startup scheduling."""

    learner_pg_bundles: list[dict[str, float]]
    learner_pg_strategy: str
    learner_pg_total_gpus: float
    learner_pg_total_cpus: float
    separate_vllm_total_gpus: float
    separate_vllm_total_cpus: float
    data_prep_actor_cpus: float
    min_total_cluster_gpus: float
    min_total_cluster_cpus: float


def format_resource_amount(amount: float) -> str:
    amount = float(amount)
    if amount.is_integer():
        return str(int(amount))
    return f"{amount:g}"


def format_resource_snapshot(resources: dict[str, float] | None) -> str:
    if not resources:
        return "{}"
    formatted = []
    for key in sorted(resources):
        value = resources[key]
        if isinstance(value, (int, float)):
            formatted.append(f"{key}={format_resource_amount(value)}")
        else:
            formatted.append(f"{key}={value}")
    return "{" + ", ".join(formatted) + "}"


def build_grpo_fast_resource_plan(args: ExperimentConfigLike, vllm_config: VLLMConfigLike) -> GrpoFastResourcePlan:
    learner_pg_bundles = [
        {"GPU": actor_num_gpus, "CPU": actor_num_gpus * LEARNER_PLACEMENT_GROUP_CPU_PER_GPU}
        for actor_num_gpus in args.num_learners_per_node
    ]
    learner_pg_total_gpus = float(sum(bundle["GPU"] for bundle in learner_pg_bundles))
    learner_pg_total_cpus = float(sum(bundle["CPU"] for bundle in learner_pg_bundles))

    # In single_gpu_mode, vLLM shares the learner placement group instead of reserving a separate one.
    separate_vllm_total_gpus = 0.0
    separate_vllm_total_cpus = 0.0
    if not args.single_gpu_mode:
        separate_vllm_total_gpus = float(vllm_config.vllm_num_engines * vllm_config.vllm_tensor_parallel_size)
        separate_vllm_total_cpus = float(vllm_config.vllm_num_engines * vllm_config.vllm_tensor_parallel_size)

    data_prep_actor_cpus = float(DATA_PREPARATION_ACTOR_NUM_CPUS)
    return GrpoFastResourcePlan(
        learner_pg_bundles=learner_pg_bundles,
        learner_pg_strategy="STRICT_SPREAD",
        learner_pg_total_gpus=learner_pg_total_gpus,
        learner_pg_total_cpus=learner_pg_total_cpus,
        separate_vllm_total_gpus=separate_vllm_total_gpus,
        separate_vllm_total_cpus=separate_vllm_total_cpus,
        data_prep_actor_cpus=data_prep_actor_cpus,
        min_total_cluster_gpus=learner_pg_total_gpus + separate_vllm_total_gpus,
        min_total_cluster_cpus=learner_pg_total_cpus + separate_vllm_total_cpus + data_prep_actor_cpus,
    )


def format_grpo_fast_resource_plan(plan: GrpoFastResourcePlan) -> str:
    lines = [
        (
            "Learner placement group "
            f"strategy={plan.learner_pg_strategy}, bundles={plan.learner_pg_bundles}, "
            f"totals=(GPU={format_resource_amount(plan.learner_pg_total_gpus)}, "
            f"CPU={format_resource_amount(plan.learner_pg_total_cpus)})"
        ),
        (
            "Separate vLLM minimum totals="
            f"(GPU={format_resource_amount(plan.separate_vllm_total_gpus)}, "
            f"CPU={format_resource_amount(plan.separate_vllm_total_cpus)})"
        ),
        (
            "Minimum full-topology totals="
            f"(GPU={format_resource_amount(plan.min_total_cluster_gpus)}, "
            f"CPU={format_resource_amount(plan.min_total_cluster_cpus)}; "
            f"includes DataPreparationActor CPU={format_resource_amount(plan.data_prep_actor_cpus)})"
        ),
    ]
    return "\n".join(lines)


def get_grpo_fast_resource_shortfalls(plan: GrpoFastResourcePlan, cluster_resources: dict[str, float]) -> list[str]:
    cluster_gpus = float(cluster_resources.get("GPU", 0.0))
    cluster_cpus = float(cluster_resources.get("CPU", 0.0))

    shortfalls = []
    if cluster_gpus < plan.learner_pg_total_gpus:
        shortfalls.append(
            "learner placement group requires "
            f"GPU={format_resource_amount(plan.learner_pg_total_gpus)} but Ray currently sees "
            f"GPU={format_resource_amount(cluster_gpus)}"
        )
    elif cluster_gpus < plan.min_total_cluster_gpus:
        shortfalls.append(
            "full topology requires at least "
            f"GPU={format_resource_amount(plan.min_total_cluster_gpus)} but Ray currently sees "
            f"GPU={format_resource_amount(cluster_gpus)}"
        )

    if cluster_cpus < plan.learner_pg_total_cpus:
        shortfalls.append(
            "learner placement group requires "
            f"CPU={format_resource_amount(plan.learner_pg_total_cpus)} but Ray currently sees "
            f"CPU={format_resource_amount(cluster_cpus)}"
        )
    elif cluster_cpus < plan.min_total_cluster_cpus:
        shortfalls.append(
            "full topology requires at least "
            f"CPU={format_resource_amount(plan.min_total_cluster_cpus)} but Ray currently sees "
            f"CPU={format_resource_amount(cluster_cpus)}"
        )

    return shortfalls
