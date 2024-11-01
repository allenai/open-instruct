from datasets import load_dataset

full_ds = load_dataset("allenai/tulu-v.3.8-mix-preview-noncommercial")

conversations = set()
prompts = set()

for elem in full_ds["train"]:
    conv = ""
    prompt = elem["messages"][0]["content"]
    prompts.add(prompt)
    for msg in elem["messages"]:
        conv += msg["content"]
    conversations.add(conv)

    ### Not using anymore:
    # ai2-adapt-dev/wildchat_gpt4_converted: 100000
    #   # ai2-adapt-dev/tulu_v3.8_unused_wildchat_prompts
    #   # ai2-adapt-dev/tulu_v3.8_unused_wildchat_conversations

seed = 42

### splitting:

# wildchat_gpt4_converted_safety_decontaminated: 100000
wildchat_ds = load_dataset("ai2-adapt-dev/wildchat_gpt4_converted_safety_decontaminated").shuffle(seed)
wildchat_ds_to_use = wildchat_ds["train"].select(range(100000))
wildchat_ds_to_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_wildchat_100k")
wildchat_ds_to_not_use = wildchat_ds["train"].select(range(100000, len(wildchat_ds["train"])))
wildchat_ds_to_not_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_wildchat_unused")

del wildchat_ds
del wildchat_ds_to_use
del wildchat_ds_to_not_use

# ai2-adapt-dev/open_math_2_gsm8k_converted: 50000
openmath2_gsm8k_ds = load_dataset("ai2-adapt-dev/open_math_2_gsm8k_converted").shuffle(seed)
openmath2_gsm8k_to_use = openmath2_gsm8k_ds["train"].select(range(50000))
openmath2_gsm8k_to_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k")
openmath2_gsm8k_to_not_use = openmath2_gsm8k_ds["train"].select(range(50000, len(openmath2_gsm8k_ds["train"])))
openmath2_gsm8k_to_not_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_unused")

del openmath2_gsm8k_ds
del openmath2_gsm8k_to_use
del openmath2_gsm8k_to_not_use

# ai2-adapt-dev/personahub_math_interm_algebra_50000: 20000
p_math_alg_ds = load_dataset("ai2-adapt-dev/personahub_math_interm_algebra_50000").shuffle(seed)
p_math_alg_ds_to_use = p_math_alg_ds["train"].select(range(20000))
p_math_alg_ds_to_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k")
p_math_alg_ds_to_not_use = p_math_alg_ds["train"].select(range(20000, len(p_math_alg_ds["train"])))
p_math_alg_ds_to_not_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_unused")

del p_math_alg_ds
del p_math_alg_ds_to_use
del p_math_alg_ds_to_not_use

# ai2-adapt-dev/processed_wildjailbreak_safety_decontaminated: 50000
wjb_ds = load_dataset("ai2-adapt-dev/processed_wildjailbreak_safety_decontaminated").shuffle(seed)
wjb_ds_to_use = wjb_ds["train"].select(range(50000))
wjb_ds_to_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k")
wjb_ds_to_not_use = wjb_ds["train"].select(range(50000, len(wjb_ds["train"])))
wjb_ds_to_not_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_unused")

del wjb_ds
del wjb_ds_to_use
del wjb_ds_to_not_use

# ai2-adapt-dev/synthetic_finalresp_wildguardmixtrain_safety_decontaminated: 50000
wg_ds = load_dataset("ai2-adapt-dev/synthetic_finalresp_wildguardmixtrain_safety_decontaminated").shuffle(seed)
wg_ds_to_use = wg_ds["train"].select(range(50000))
wg_ds_to_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k")
wg_ds_to_not_use = wg_ds["train"].select(range(50000, len(wg_ds["train"])))
wg_ds_to_not_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_unused")

del wg_ds
del wg_ds_to_use
del wg_ds_to_not_use

# ai2-adapt-dev/sciriff_converted: 10000
sciriff_ds = load_dataset("ai2-adapt-dev/sciriff_converted").shuffle(seed)
sciriff_ds_to_use = sciriff_ds["train"].select(range(10000))
sciriff_ds_to_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_sciriff_10k")
sciriff_ds_to_not_use = sciriff_ds["train"].select(range(10000, len(sciriff_ds["train"])))
sciriff_ds_to_not_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_sciriff_unused")

del sciriff_ds
del sciriff_ds_to_use
del sciriff_ds_to_not_use

# ai2-adapt-dev/table_gpt_converted: 5000
table_gpt_ds = load_dataset("ai2-adapt-dev/table_gpt_converted").shuffle(seed)
table_gpt_ds_to_use = table_gpt_ds["train"].select(range(5000))
table_gpt_ds_to_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_table_gpt_5k")
table_gpt_ds_to_not_use = table_gpt_ds["train"].select(range(5000, len(table_gpt_ds["train"])))
table_gpt_ds_to_not_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_table_gpt_unused")

del table_gpt_ds
del table_gpt_ds_to_use
del table_gpt_ds_to_not_use

# ai2-adapt-dev/aya_dataset_converted: 100000
aya_ds = load_dataset("ai2-adapt-dev/aya_dataset_converted").shuffle(seed)
aya_ds_to_use = aya_ds["train"].select(range(100000))
aya_ds_to_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_aya_100k")
aya_ds_to_not_use = aya_ds["train"].select(range(100000, len(aya_ds["train"])))
aya_ds_to_not_use.push_to_hub("ai2-adapt-dev/tulu_v3.9_aya_unused")

del aya_ds
del aya_ds_to_use
del aya_ds_to_not_use
