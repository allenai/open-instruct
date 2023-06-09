"""
Kick off job to compress a smaller model so that we don't have to debug the huge one.
"""

import beaker
from beaker import Beaker, ExperimentSpec, TaskSpec

beaker_client = Beaker.from_env(default_workspace="ai2/davidw")

wkdir = "$NFS_HOME/proj/open-instruct/quantize"
python_cmd = (
    "python quantize_autogptq_wikitext.py "
    "--pretrained_model_dir /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B "
    "--quantized_model_dir /net/nfs.cirrascale/allennlp/davidw/checkpoints/gptq_llama_7b"
)

spec = ExperimentSpec(
    description="GPTQ quantization.",
    tasks=[
        TaskSpec(
            name="autogptq_llama_7b",
            image=beaker.ImageSource(beaker="01GZHG16S90N033XP4D6BPC8NR"),
            command=["bash", "-c", f"cd {wkdir}; {python_cmd}"],
            result=beaker.ResultSpec(
                path="/unused"  # required even if the task produces no output.
            ),
            datasets=[
                beaker.DataMount(
                    source=beaker.DataSource(host_path="/net/nfs.cirrascale"),
                    mount_path="/net/nfs.cirrascale",
                )
            ],
            context=beaker.TaskContext(priority=beaker.Priority("high")),
            constraints=beaker.Constraints(
                cluster=["ai2/s2-cirrascale", "ai2/allennlp-cirrascale"]
            ),
            env_vars=[
                beaker.EnvVar(
                    name="NFS_HOME", value="/net/nfs.cirrascale/allennlp/davidw"
                ),
                beaker.EnvVar(
                    name="HF_HOME",
                    value="/net/nfs.cirrascale/allennlp/davidw/cache/huggingface"
                ),
            ],
            resources=beaker.TaskResources(gpu_count=1),
        ),
    ],
)

experiment_name = "quantize"
workspace_name = "ai2/davidw"

experiment = beaker_client.experiment.create(
    experiment_name,
    spec,
    workspace=workspace_name,
)
