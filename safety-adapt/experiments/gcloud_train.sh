gcloud alpha compute tpus tpu-vm ssh davidw-v3-256 --zone=us-east1-d --project=ai2-tpu --worker=all --command="cd easylm; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=50 \
    --save_model_freq=1000 \
    --save_milestone_freq=0 \
    --load_llama_config='7b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/7b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=2e-5 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=32 \
    --train_dataset.type='tulu_json_torch' \
    --train_dataset.json_torch_dataset.path='gs://davidw-dev/science-adapt/data/4k_balance_task_10k.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=4096 \
    --train_dataset.json_torch_dataset.batch_size=4 \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='open_instruct' \
    --logger.output_dir='gs://davidw-dev/science-adapt/models/1stage_4k_balance_task_10k' &> all.log &"




gcloud alpha compute tpus tpu-vm ssh davidw-v3-256 --zone=us-east1-d --project=ai2-tpu --command="cat easylm/all.log"
