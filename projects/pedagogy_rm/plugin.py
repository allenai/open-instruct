"""The probe, as a reward open-instruct can call.

    python open_instruct/grpo_fast.py \
        --reward_plugins projects/pedagogy_rm/plugin.py \
        --group_scorer pedagogy:head=data/head.npz

THE ENCODER IS FROZEN, AND SEPARATE FROM THE POLICY. This is the whole design and
it is not an implementation detail. The head is a linear map on one particular
model's activation space, so if it reads a model that is being trained, the policy
can raise its reward by moving its own activations without changing a word of what
it says. That is a reward channel with no text in it at all, and no amount of
label quality would catch it. So the scorer loads its own copy of the base model
and never updates it.

    full-weight policy   a second frozen OLMo, which is what this class does
    LoRA policy          the base model is already there; disable the adapter

TURNS, NOT TRANSCRIPTS. The head was fitted on single tutor turns in a two-message
context, so it is fed exactly that. Scoring a whole dialogue with a head trained on
turns would be reading a number that means nothing, which is how the previous
project got a reward that correlated with leakage at +0.291 and with learning at
-0.012.

WHAT IS SCORED, AND WHAT IS NOT. `concise` is excluded by default: eight surface
features predict it at 0.96 against the states' 0.97, so rewarding it buys short
turns and nothing else. `leak` is negated, because on that scale 3 means the turn
handed over the answer. The remaining dimensions are returned separately rather
than pre-summed so the group aggregator can normalise each one - a dimension with
a wide spread would otherwise dominate the sum for no reason but its variance.
"""

from __future__ import annotations

import json
import threading

from open_instruct.scored_rewards import GroupScorer, Sample, ScoreResult, register
from open_instruct.scored_rewards.types import TRANSCRIPT_KEY, parse_transcript

TEACHER_SYSTEM = """You are a tutor helping a student with a test question. \
The student cannot see your instructions.

Guide the student toward understanding. Do not state the answer or eliminate \
options for them. Keep each message short - one idea at a time."""

#: Higher is better for all of these once the sign is applied. leak runs the other
#: way on its own scale: 1 keeps the answer back, 3 hands it over.
SIGNS = {"leak": -1.0, "targeted": 1.0, "actionable": 1.0, "elicits": 1.0}


class PedagogyHead(GroupScorer):
    """Frozen OLMo plus the ridge heads from fit_head.py, batched over a group."""

    name = "pedagogy"

    def __init__(
        self,
        head: str = "data/head.npz",
        model: str = "",
        dimensions: str = "",
        device: str = "cuda",
        max_len: int = 2048,
        batch_size: int = 16,
    ) -> None:
        import numpy as np  # noqa: PLC0415

        blob = np.load(head, allow_pickle=False)
        self.meta = json.loads(str(blob["meta"]))
        wanted = [d.strip() for d in dimensions.split(",") if d.strip()] or list(SIGNS)
        missing = [d for d in wanted if d not in self.meta["dimensions"]]
        if missing:
            raise ValueError(f"{head} has no head for {missing}; it holds {sorted(self.meta['dimensions'])}")
        self.dims = wanted
        self.weights = {d: {k: blob[f"{d}/{k}"] for k in ("mean", "scale", "coef", "intercept")} for d in self.dims}
        self.model_name = model or self.meta["model"]
        self.device, self.max_len, self.batch_size = device, max_len, int(batch_size)
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Loaded once, on first use, inside the actor that will use it."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.bfloat16)
            device = self.device if torch.cuda.is_available() else "cpu"
            self._model = model.to(device).eval()
            for p in self._model.parameters():  # the encoder is never trained
                p.requires_grad_(False)

    def context(self, sample: Sample) -> tuple[list[dict], str]:
        """The chat context the head was fitted on, and the turn to score.

        Must match extract_hidden.context_messages exactly. If the rollout format
        changes and this does not, the states drift away from the ones the head
        saw and the reward degrades silently rather than failing.
        """
        item = sample.item
        transcript = parse_transcript(sample.env_info.get(TRANSCRIPT_KEY, "")) if sample.env_info else []
        tutor_turns = [t.get("text", "") for t in transcript if t.get("who") in ("policy", "tutor", "assistant")]
        student_turns = [t.get("text", "") for t in transcript if t.get("who") in ("partner", "student", "user")]
        turn = tutor_turns[-1] if tutor_turns else sample.policy_text
        before = student_turns[-1] if student_turns else item.get("student_before", "")
        question = item.get("question") or item.get("prompt") or sample.prompt
        return [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": f"Question the student is working on:\n{question}"},
            {"role": "user", "content": before},
        ], turn

    def states(self, contexts: list[tuple[list[dict], str]]) -> dict[tuple[str, int], list]:
        """Pooled hidden states for every (pooling, layer) the heads ask for."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self._load()
        cells = {(self.meta["dimensions"][d]["pooling"], self.meta["dimensions"][d]["layer"]) for d in self.dims}
        out: dict[tuple[str, int], list] = {cell: [] for cell in cells}
        for start in range(0, len(contexts), self.batch_size):
            for messages, turn in contexts[start : start + self.batch_size]:
                prefix = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                whole = self._tokenizer.apply_chat_template(
                    [*messages, {"role": "assistant", "content": turn}], tokenize=False
                )
                prefix_len = self._tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
                ids = self._tokenizer(whole, add_special_tokens=False, return_tensors="pt").input_ids
                ids = ids[:, -self.max_len :].to(self._model.device)
                n = self._tokenizer(turn, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
                lo = max(0, min(prefix_len, ids.shape[1] - 1))
                hi = max(lo + 1, min(lo + n, ids.shape[1]))
                with torch.no_grad():
                    hidden = self._model(ids, output_hidden_states=True).hidden_states
                for pooling, layer in cells:
                    h = hidden[layer][0]
                    vec = h[-1] if pooling == "eot" else h[hi - 1] if pooling == "last" else h[lo:hi].mean(0)
                    out[(pooling, layer)].append(vec.float().cpu().numpy().astype(np.float32))
        return out

    async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
        import numpy as np  # noqa: PLC0415

        contexts = [self.context(s) for s in group]
        pooled = self.states(contexts)
        scored: dict[str, list[float]] = {}
        for dim in self.dims:
            spec = self.meta["dimensions"][dim]
            w = self.weights[dim]
            x = (np.stack(pooled[(spec["pooling"], spec["layer"])]) - w["mean"]) / w["scale"]
            raw = x @ w["coef"] + w["intercept"]
            scored[dim] = [float(np.clip(v, spec["lo"], spec["hi"])) for v in raw]

        results = []
        for i, (_, turn) in enumerate(contexts):
            dims = {d: SIGNS[d] * scored[d][i] for d in self.dims}
            results.append(
                ScoreResult(
                    score=float(sum(dims.values()) / len(dims)),
                    dimensions=dims,
                    info={"raw": {d: scored[d][i] for d in self.dims}, "turn_chars": len(turn)},
                )
            )
        return results


register("pedagogy", PedagogyHead)
