#!/bin/bash

: '
Example usage:

BEAKER_TOKEN=xxxx \
HF_TOKEN=xxxx \
WANDB_API_KEY=xxxx \
OPENAI_API_KEY=xxxx \
ACCOUNT=costah \
basedir=/weka/oe-adapt-default \
sh setup.sh
'

# Create user directory
basedir=${basedir:-/weka/oe-adapt-default}
ACCOUNT=${ACCOUNT:-$(whoami)}
mkdir -p $basedir/$ACCOUNT

# Setup persistent bashrc
PERSISTENT_BASHRC_PATH=$basedir/$ACCOUNT/persistent_bashrc

# Write environment setup
cat << EOF > $PERSISTENT_BASHRC_PATH
# Credentials
export BEAKER_TOKEN=$BEAKER_TOKEN
export HF_TOKEN=$HF_TOKEN
export WANDB_API_KEY=$WANDB_API_KEY
export OPENAI_API_KEY=$OPENAI_API_KEY
export ACCOUNT=$ACCOUNT
export basedir=$basedir

# Environment exports
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_DATASETS_CACHE=\$basedir/allennlp/.cache/huggingface
export HF_HUB_CACHE=\$basedir/allennlp/.cache/hub
export HF_ASSETS_CACHE=\$basedir/allennlp/.cache/assets
export UV_CACHE_DIR=\$basedir/\$ACCOUNT/.cache/uv/
export UV_INSTALL_DIR=\$basedir/\$ACCOUNT/uv/
export UV_PYTHON_INSTALL_DIR=\$basedir/\$ACCOUNT/uv/python/

# Pyenv setup
export PYENV_ROOT="\$basedir/\$ACCOUNT/pyenv"
export PATH="\$PYENV_ROOT/bin:\$PATH"
if [ ! -d "\$PYENV_ROOT" ]; then
    curl -s https://pyenv.run | bash
fi
eval "\$(pyenv init -)"
eval "\$(pyenv virtualenv-init -)"
source \$(pyenv root)/completions/pyenv.bash
unset CONDA_ENVS_DIRS
EOF

# Add to .bashrc
echo "source $PERSISTENT_BASHRC_PATH" >> ~/.bashrc
echo "added 'source $PERSISTENT_BASHRC_PATH' to ~/.bashrc"

echo "🔥 Setup complete! Use this in new sessions (save it somewhere):
source $PERSISTENT_BASHRC_PATH"