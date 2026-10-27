"""Reversible instrumentation controls for the fixed-token numerical probe."""

import json

import torch
from miles.backends.core_utils import vision_models
from olmo_core.nn.moe.v2 import olmo3
from PIL import Image


@torch.no_grad()
def core_controls(model, visual, module, hf, processor, sample, output):
    tokens = torch.tensor(sample["tokens"], device="cuda", dtype=torch.long)
    n = sample["response_length"]
    batch = {
        "tokens": tokens[None],
        "response_lengths": [n],
        "multimodal_train_inputs": [{k: v.to("cuda") for k, v in sample["multimodal_train_inputs"].items()}],
    }
    combined = vision_models.conditioning_inputs(module, batch)["input_embeddings"]
    images = [Image.open(item["path"]).convert("RGB") for item in sample["metadata"]["vision"]["images"]]
    _, prepared = processor.prepare(sample["metadata"]["vision"]["prepared"]["placeholder_token_ids"], images)
    features = torch.cat(
        [visual.encode_images(p.patches.to("cuda", torch.bfloat16), p.pooling.to("cuda")) for p in prepared], dim=1
    )
    separate = model.forward_embed(tokens[None]).index_add(
        1, batch["multimodal_train_inputs"][0]["image_positions"], features
    )
    inputs = {"combined_images": combined, "separate_images": separate}
    report = {}
    captured = {}

    def capture(name):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(value, torch.Tensor):
                captured[name] = value.detach().clone()

        return hook

    def score(label, trace=False):
        report[label] = {}
        for variant, embeddings in inputs.items():
            captured.clear()
            logits = model(tokens[None], input_embeddings=embeddings, loss_div_factor=tokens.numel())
            scores = logits[0, -n - 1 : -1].float().log_softmax(-1).gather(1, tokens[-n:, None]).flatten()
            report[label][variant] = scores.tolist()
            if trace:
                torch.save(
                    {k: v.cpu() for k, v in {"input_embeddings": embeddings, **captured}.items()},
                    output.with_name(f"control-{label}-{variant}.pt"),
                )
        output.with_name("core-controls.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"control": label, "scores": report[label]}), flush=True)

    score("before_hooks")
    handles = [b.register_forward_hook(capture(f"layer_{i}")) for i, b in enumerate(model.blocks.values())]
    handles.append(model.lm_head.norm.register_forward_hook(capture("head_input")))
    score("layer_hooks", trace=True)
    first = next(iter(model.blocks.values()))
    detail = [m.register_forward_hook(capture("detail." + name)) for name, m in first.named_modules() if name]
    score("detail_hooks", trace=True)
    for handle in handles + detail:
        handle.remove()
    captured.clear()
    score("after_hooks")
    # Reproduce the additional export work separately from hooks, retaining the
    # last exported tensor just as the original audit did.
    for _, value in olmo3.iter_olmo3_moe_hf_state(model, hf.text_config):
        exported = value.detach().cpu()
        torch.equal(exported.float(), exported.bfloat16().float())
    score("after_export")
    return report
