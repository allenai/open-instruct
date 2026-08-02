"""A learned scalar head on a frozen backbone.

The other kind of score-based reward: instead of asking a model to grade in
words, fit a head on human ratings and read a number off it. Cheap per call,
dense, and the only option when the quality you care about is not something a
prompt can describe.

It is also the easiest reward to fool, so the defaults here are the
conservative ones, each for a measured reason.

FROZEN backbone. RL moves the policy every step. A head reading a trunk that is
itself being trained scores a moving target.

LAST POSITION, not mean-pooled. Attention is causal, so only the final token has
read the whole text. Mean-pooling weights the beginning.

LINEAR head, not an MLP. In the run this came from, the MLP reached train
Spearman 0.976 against test 0.55 - memorisation - with AUC swinging 0.87-0.94
across splits, while the linear probe held AUC 0.93 with a third of the
variance. It also extrapolates predictably, which is the property that matters:
RL will push the policy into regions the head never saw, and a single direction
in embedding space degrades gracefully there where an MLP has arbitrary maxima
to climb.

READ THIS BEFORE YOU USE IT. A head fitted on ratings from two sources can learn
to recognise the SOURCE rather than the quality - which tier a sample came from
is often easier to predict than whether it is good. Check two things before
letting one drive a run: that tier is NOT recoverable from the embedding, and
that within-group ranking (the only comparison the RL update ever makes) holds
up on the tier your policy actually occupies. A global AUC of 0.92 can coexist
with a within-group ranking barely above a coin flip, because the global number
pools comparisons the algorithm never makes.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence

from open_instruct.scored_rewards.types import Sample, Scorer, ScoreResult

#: Checkpoint schema. ``mu``/``sd`` are the standardisation of the backbone's
#: final hidden state, fitted on the training set and saved with the head so
#: scoring cannot silently drift from fitting.
CHECKPOINT_KEYS = ("backbone", "head", "mu", "sd")


class LinearHead:
    """Frozen backbone plus a linear head, batched."""

    def __init__(self, checkpoint: str, device: str = "cuda", batch_size: int = 16, max_length: int = 1024):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"reward head not found: {checkpoint}")
        blob = torch.load(checkpoint, map_location="cpu")
        missing = [k for k in CHECKPOINT_KEYS if k not in blob]
        if missing:
            raise ValueError(f"{checkpoint} is missing {missing}; expected keys {CHECKPOINT_KEYS}")

        self.torch = torch
        self.mu, self.sd = blob["mu"], blob["sd"]
        self.device, self.batch_size, self.max_length = device, batch_size, max_length

        self.tokenizer = AutoTokenizer.from_pretrained(blob["backbone"])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(blob["backbone"], output_hidden_states=True).to(device).eval()
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        state = blob["head"]
        out_features, in_features = state["weight"].shape
        self.head = torch.nn.Linear(in_features, out_features)
        self.head.load_state_dict(state)
        self.head.to(device).eval()
        self.column = out_features - 1

    def __call__(self, texts: Sequence[str]) -> list[float]:
        torch = self.torch
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = list(texts[start : start + self.batch_size])
                enc = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    padding_side="left",
                ).to(self.device)
                hidden = self.model(**enc).hidden_states[-1][:, -1, :].float()
                z = (hidden.cpu() - self.mu) / self.sd
                out.extend(self.head(z.to(self.head.weight.device))[:, self.column].tolist())
        return out


class HeadScorer(Scorer):
    """A ``LinearHead`` as a per-sample scorer.

    ``view`` renders a sample into the exact string the head was FITTED on.
    Getting this wrong is the failure mode: a head fitted on one framing and
    applied to another is scoring text it has never seen, and nothing about the
    output will look wrong.

    The output is an uncalibrated RANKING score, not a rating. Fitting the
    absolute scale buys nothing here, because the reward normalises within the
    group and only the ordering survives - wrap this in
    ``guards.MultiDimensional`` or let ``GroupRewardConfig`` centre it.
    """

    def __init__(
        self,
        head: LinearHead | Callable[[Sequence[str]], list[float]],
        view: Callable[[Sample], str],
        name: str = "head",
    ):
        self.head = head
        self.view = view
        self.name = name

    def score_sync(self, sample: Sample) -> ScoreResult:
        value = self.head([self.view(sample)])[0]
        return ScoreResult(score=float(value), info={f"{self.name}_raw": float(value)})

    async def score_batch(self, samples: Sequence[Sample]) -> list[ScoreResult]:
        """One backbone pass for the whole group instead of G passes."""
        views = [self.view(s) for s in samples]
        values = await asyncio.get_event_loop().run_in_executor(None, self.head, views)
        return [ScoreResult(score=float(v), info={f"{self.name}_raw": float(v)}) for v in values]
