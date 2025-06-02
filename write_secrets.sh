workspace=ai2/oe-adapt-code
user=saurabhs
for secret in HF_TOKEN WANDB_API_KEY BEAKER_TOKEN OPENAI_API_KEY AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AZURE_API_KEY; do
  command="beaker secret write ${user}_${secret} ${!secret} -w $workspace"
  echo $command
  eval $command
done