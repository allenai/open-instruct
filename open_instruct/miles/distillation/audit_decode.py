"""Capture selected prediction positions without changing sampled tokens."""

import json
from pathlib import Path

import torch

SESSIONS = {}


def last_row(value, rows):
    if not isinstance(value, torch.Tensor) or value.numel() % rows:
        raise ValueError("Expected a tensor with one row per input position")
    return value.detach().reshape(rows, -1)[-1].cpu().clone()


def factory(config):
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    state = SESSIONS.setdefault(str(root), {})
    installed = False

    def before(model, args):
        state.clear()
        control = root / "control.json"
        if not control.exists():
            return
        settings = json.loads(control.read_text())
        positions = args[1]
        position = int(positions[-1])
        if position in settings["positions"]:
            state.update(
                position=position,
                label=settings["label"],
                rows=positions.numel(),
                input_token=int(args[0][-1]),
                states={},
            )

    def after(model, args, output):
        nonlocal installed
        if installed:
            return output
        model.register_forward_pre_hook(before)
        for name, module in model.named_modules():
            boundary = name.startswith("layers.") and name.count(".") in (1, 2)
            detail = name.startswith("layers.0.self_attn.")
            if not (boundary or detail or name in {"embed_norm", "norm"}):
                continue

            def capture(module, args, output, name=name):
                if state:
                    value = output[0] if isinstance(output, tuple) else output
                    state["states"][name] = last_row(value, state["rows"])

            module.register_forward_hook(capture)
            if name.endswith("o_norm"):

                def capture_input(module, args, name=name):
                    if state:
                        state["states"][name + ".input"] = last_row(args[0], state["rows"])

                module.register_forward_pre_hook(capture_input)
        installed = True
        return output

    return after


def head_factory(config):
    root = Path(config["root"])
    state = SESSIONS.setdefault(str(root), {})

    def after(module, args, output):
        if state:
            if output.next_token_logits is None or output.next_token_logits.shape[0] != 1:
                raise ValueError("Decode audit requires one request and next-token logits")
            state["logits"] = output.next_token_logits[0].detach().cpu().clone()
            torch.save(state, root / f"{state['label']}-{state['position']}.pt")
            state.clear()
        return output

    return after
