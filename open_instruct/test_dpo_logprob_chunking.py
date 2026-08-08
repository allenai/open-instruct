"""The chunked DPOLMHead must reproduce the unchunked per-token logprob computation exactly
(values and gradients) — the chunking exists purely to avoid materializing full-vocab logits
(measured 110 GiB OOM at 32k seq x 248k vocab on Qwen3.5-9B)."""

import torch
from torch import nn

from open_instruct import olmo_core_train_modules, padding_free_collator
from open_instruct.olmo_core_train_modules import DPOLMHead


class _TestHead(DPOLMHead):
    def __init__(self, d_model: int, vocab_size: int):
        nn.Module.__init__(self)
        self.norm = None
        self.w_out = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def tp_enabled(self) -> bool:
        return False


def _reference(head: _TestHead, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return padding_free_collator.calculate_per_token_logps(head.w_out(h), labels)


def _make_inputs(seq_len: int, d_model: int = 32, batch: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    h = torch.randn(batch, seq_len, d_model, requires_grad=True)
    labels = torch.randint(0, 128, (batch, seq_len))
    labels[:, : seq_len // 3] = -100  # masked prefix, as in real rows
    return h, labels


def test_chunked_matches_reference_values_and_grads(monkeypatch):
    monkeypatch.setattr(olmo_core_train_modules, "DPO_LOGPROB_CHUNK_SIZE", 7)  # force ragged chunks
    head = _TestHead(32, 128)
    h, labels = _make_inputs(seq_len=23)

    out = head(h, labels=labels)
    chunked = out.loss
    h_ref = h.detach().clone().requires_grad_(True)
    reference = _reference(head, h_ref, labels)
    torch.testing.assert_close(chunked, reference, atol=1e-6, rtol=1e-6)

    chunked.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(h.grad, h_ref.grad, atol=1e-6, rtol=1e-6)


def test_chunked_matches_reference_no_grad(monkeypatch):
    monkeypatch.setattr(olmo_core_train_modules, "DPO_LOGPROB_CHUNK_SIZE", 8)
    head = _TestHead(32, 128)
    h, labels = _make_inputs(seq_len=16, seed=1)
    with torch.no_grad():
        chunked = head(h, labels=labels).loss
        reference = _reference(head, h, labels)
    torch.testing.assert_close(chunked, reference, atol=1e-6, rtol=1e-6)
