# Two-engine concurrency-32 follow-up

Compare the qualified 2T/4I, concurrency-8, batch-128 configuration with
`steady-2t2i-c32-b128-graphs`: 2 trainer GPUs and 2 TP1 inference engines.
Each engine permits 32 outstanding HTTP requests and 32 running sequences;
full decode CUDA graphs capture through batch 32, with prefill graphs disabled.
Token and recurrent-state pools increase from 131072/128 to 262144/256.

The model, prepared GSM8K fixture, objective, optimization batch 128, automatic
producer budget 128, completed FIFO capacity 128 samples, whole-group dispatch,
lag two, publication, and diagnostics match the preceding batch-128 trial.
No evaluation, save or export stages are included in the timed exercise.

Run 16 updates, exclude the first six provisionally, then inspect phase timings
for late warmup effects. Require all-rank optimizer and route-replay audits,
complete token accounting, and clean workflow shutdown. Compare useful tokens/s,
queue waiting, handoff time, drop fractions/length/age, active sequences, GPU
activity and memory. Hardware warp occupancy and achieved bandwidth are not
measured by the retained NVML sampler.

Status: prepared; measurements will be added after the trial completes.
