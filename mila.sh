# set in your bashrc
# HF_HOME=/network/scratch/n/noukhovm/huggingface
# create venv called env
# pip install -r requirements.txt
# export WANDB_PROJECT=trl
# export WANDB_ENTITY=mila-language-drift
module load python/3.10
module load cuda/12.1.1
source .venv/bin/activate
# mkdir $SLURM_TMPDIR/$SLURM_JOB_ID
# git clone --filter=blob:none --no-checkout $(GITHUB_REPO) $(_WORKDIR)
# git clone
# mkdir -p results/
