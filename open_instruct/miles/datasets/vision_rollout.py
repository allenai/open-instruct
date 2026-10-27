"""Single-turn Olmo image generation through MILES' existing rollout lifecycle.

Configure custom_generate_function_path to this module's ``generate``. It sends
unexpanded placeholders and original PNG bytes to serving, while recording the
independently validated expanded IDs and image tensors for trainer scoring.
Async refresh, partial continuations and router replay remain explicit follow-ons.
"""

import base64
import functools
from pathlib import Path

from miles.rollout.generate_utils import generate_endpoint_utils
from miles.utils import http_utils, types

from miles.rollout import base_types
from open_instruct.miles.datasets import vision, vision_replay


@functools.lru_cache(maxsize=2)
def processor(checkpoint):
    return vision.load_processor(Path(checkpoint))


async def generate(request: base_types.GenerateFnInput) -> base_types.GenerateFnOutput:
    args, sample = request.args, request.sample
    if sample.response_length or sample.response:
        raise ValueError("Olmo image partial continuations are not yet qualified")
    if args.use_rollout_routing_replay or args.use_rollout_indexer_replay:
        raise ValueError("Olmo image routing/indexer replay is not yet qualified")
    row = {
        "input": sample.prompt,
        "images": [item["path"] for item in sample.metadata["vision"]["images"]],
        "metadata": sample.metadata,
    }
    if "prepared" not in row["metadata"]["vision"]:
        raise ValueError("Image rows must pass immutable preparation before rollout")
    image_processor = processor(args.hf_checkpoint)
    ids, images = vision.process_row(row, image_processor, {})
    if args.rollout_max_prompt_len is not None and len(ids) > args.rollout_max_prompt_len:
        raise ValueError("Expanded image prompt exceeds rollout_max_prompt_len")
    params = dict(request.sampling_params)
    requested = params.get("max_new_tokens", args.rollout_max_response_len)
    params["max_new_tokens"] = requested
    available = args.rollout_max_context_len - len(ids) if args.rollout_max_context_len else requested
    if available < requested:
        raise ValueError("Image-expanded context cannot fit the requested response budget")
    sample.tokens = ids
    sample.multimodal_train_inputs = vision_replay.training_inputs(images, ids, image_processor.token_ids.im_patch_id)
    sample.multimodal_inputs = {"images": row["images"]}
    # The standard MILES transport carries train_metadata, not the verifier-only metadata.
    sample.train_metadata = {
        key: sample.metadata[key]
        for key in ("vision", "prepared_sample_id", "pair_id", "source_image_sha256")
        if key in sample.metadata
    }
    payload = {
        "input_ids": sample.metadata["vision"]["prepared"]["placeholder_token_ids"],
        "image_data": [
            "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")
            for path in row["images"]
        ],
        "sampling_params": params,
        "return_logprob": True,
    }
    output = await http_utils.post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        payload,
        headers=generate_endpoint_utils.compute_routing_headers(args, sample),
    )
    if output["meta_info"].get("prompt_tokens") != len(ids):
        raise ValueError("Serving and trainer disagree on image-expanded prompt length")
    # tokens is already the expanded trainer sequence, so the helper appends only response IDs.
    await generate_endpoint_utils.update_sample_from_response(args, sample, payload=payload, output=output)
    if sample.status == types.Sample.Status.COMPLETED and not sample.response_length:
        raise ValueError("Image generation completed without response tokens")
    return base_types.GenerateFnOutput(samples=sample)
