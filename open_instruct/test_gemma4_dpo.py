"""CPU tests for open_instruct/gemma4_dpo.py (Gemma-4 DPO on the HF/DeepSpeed path).

Skipped unless transformers >= 5.5 (Gemma4ForCausalLM). Run them in the Gemma env, e.g.
    /tmp/oi-venv-g4/bin/python -m pytest open_instruct/test_gemma4_dpo.py
The tokenizer tests also need the Gemma-4 servable on weka (skipped when it is absent).
"""

import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForCausalLM"):
    pytest.skip("transformers < 5.5 has no Gemma4ForCausalLM", allow_module_level=True)

from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig  # noqa: E402

from open_instruct import gemma4_dpo  # noqa: E402

SERVABLE = "/weka/oe-adapt-default/mingqianz/models/gemma4-12b-nothink-servable"


def _tiny_model(seed: int = 0, attn: str = "sdpa") -> Gemma4ForCausalLM:
    cfg = Gemma4TextConfig(
        vocab_size=384,
        vocab_size_per_layer_input=384,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        final_logit_softcapping=30.0,
        hidden_size_per_layer_input=0,
        max_position_embeddings=1024,
    )
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    model = Gemma4ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        model.model.embed_tokens.weight.mul_(150.0)  # tied lm_head: push logits into the softcap's nonlinear range
        for layer in model.model.layers:
            layer.layer_scalar.uniform_(0.3, 1.0)
    return model


def _reference_token_logps(model, input_ids, labels, **kwargs):
    """Full-logit reference via the ORIGINAL forward (HF applies the softcap) + open-instruct's per-token gather."""
    logits = model(input_ids=input_ids, **kwargs).logits.float()
    targets = torch.zeros_like(labels)
    targets[:, :-1] = labels[:, 1:]
    return torch.gather(logits.log_softmax(-1), 2, targets.clamp(min=0).unsqueeze(2)).squeeze(2)


@pytest.fixture(scope="module", autouse=True)
def _patch():
    gemma4_dpo.install_forward_patch()


def test_patch_passthrough_is_original_forward():
    model = _tiny_model()
    x = torch.randint(0, 384, (1, 20))
    with torch.no_grad():
        a = model(input_ids=x).logits
        b = gemma4_dpo._ORIG_FORWARD(model, input_ids=x).logits
    assert torch.equal(a, b)


@pytest.mark.parametrize("chunk", [1, 5, 16, 1000])
def test_chunked_softcap_logps_values_and_grads(chunk, monkeypatch):
    monkeypatch.setattr(gemma4_dpo, "LOGPROB_CHUNK_SIZE", chunk)
    L = 37
    x = torch.randint(0, 384, (1, L))
    labels = x.clone()
    labels[:, :11] = -100

    ref_model, test_model = _tiny_model(), _tiny_model()
    ref = _reference_token_logps(ref_model, x, labels)
    got = gemma4_dpo.chunked_softcap_logps(
        test_model.model(input_ids=x).last_hidden_state, labels, test_model.lm_head, 30.0, chunk_size=chunk
    )
    mask = labels[:, 1:] != -100
    assert torch.allclose(got[:, :-1][mask], ref[:, :-1][mask], atol=1e-5, rtol=0)
    # the softcap is really exercised: some raw logits exceed the cap
    with torch.no_grad():
        raw = test_model.lm_head(test_model.model(input_ids=x).last_hidden_state)
    assert raw.abs().max() > 30.0

    (ref[:, :-1][mask].sum()).backward()
    (got[:, :-1][mask].sum()).backward()
    for (n, p_ref), (_, p_got) in zip(ref_model.named_parameters(), test_model.named_parameters()):
        assert p_ref.grad is not None and p_got.grad is not None, n
        # fp32 summation order differs (chunks); bound the error relative to the tensor's largest gradient
        err = (p_got.grad - p_ref.grad).abs().max()
        assert err <= 1e-5 * p_ref.grad.abs().max() + 1e-7, (n, err, p_ref.grad.abs().max())


@pytest.mark.parametrize("n", [1, 3, 32, 50])
def test_fixed_num_chunks_values_and_grads(n):
    L = 37  # n = 50 > L exercises empty pieces
    x = torch.randint(0, 384, (1, L))
    labels = x.clone()
    labels[:, :9] = -100
    ref_model, test_model = _tiny_model(), _tiny_model()
    ref = _reference_token_logps(ref_model, x, labels)
    got = gemma4_dpo.chunked_softcap_logps(
        test_model.model(input_ids=x).last_hidden_state, labels, test_model.lm_head, 30.0, num_chunks=n
    )
    mask = labels[:, 1:] != -100
    assert got.shape == ref.shape
    assert torch.allclose(got[:, :-1][mask], ref[:, :-1][mask], atol=1e-5, rtol=0)
    (ref[:, :-1][mask].sum()).backward()
    (got[:, :-1][mask].sum()).backward()
    for (nm, p_ref), (_, p_got) in zip(ref_model.named_parameters(), test_model.named_parameters()):
        err = (p_got.grad - p_ref.grad).abs().max()
        assert err <= 1e-5 * p_ref.grad.abs().max() + 1e-7, (nm, err)


def test_dpo_labels_path_equals_reference():
    model = _tiny_model()
    x = torch.randint(0, 384, (1, 29))
    labels = x.clone()
    labels[:, :5] = -100
    with torch.no_grad():
        ref = _reference_token_logps(model, x, labels)
        got = model(input_ids=x, dpo_labels=labels)
    assert torch.allclose(got[:, :-1], ref[:, :-1], atol=1e-5, rtol=0)


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_packed_segments_equal_separate_forwards(attn):
    model = _tiny_model(attn=attn)
    a = torch.randint(0, 384, (1, 13))
    b = torch.randint(0, 384, (1, 21))
    ids = torch.cat([a, b], dim=1)
    seg = torch.tensor([1] * 13 + [2] * 21)
    labels = ids.clone()
    pos = gemma4_dpo.segment_positions(seg)
    assert pos.tolist() == list(range(13)) + list(range(21))
    with torch.no_grad():
        packed = model(
            input_ids=ids,
            attention_mask=gemma4_dpo.gemma4_masks(seg, model.config.sliding_window, attn),
            position_ids=pos[None],
            dpo_labels=labels,
        )
        sep_a = model(input_ids=a, dpo_labels=a.clone())
        sep_b = model(input_ids=b, dpo_labels=b.clone())
    assert torch.allclose(packed[:, :12], sep_a[:, :12], atol=1e-5, rtol=0)
    assert torch.allclose(packed[:, 13:-1], sep_b[:, :-1], atol=1e-5, rtol=0)


def test_separate_forward_and_batch_logps():
    model = _tiny_model()
    ids = torch.randint(0, 384, (1, 30))
    seg = torch.tensor([[1] * 10 + [2] * 16 + [0] * 4])  # 2 segments + right padding
    labels = ids.clone()
    labels[0, :12] = -100
    labels[0, 26:] = -100
    batch = {
        "chosen_input_ids": ids,
        "chosen_attention_mask": seg,
        "chosen_labels": labels,
        "rejected_input_ids": ids[:, :26],
        "rejected_attention_mask": seg[:, :26],
        "rejected_labels": labels[:, :26],
    }
    with torch.no_grad():
        c, r, aux = gemma4_dpo.gemma4_separate_forward(model, batch, average_log_prob=True)
    assert aux is None and torch.allclose(c, r, atol=1e-6)  # padding is stripped, not attended
    try:
        from open_instruct.dpo_utils import _get_batch_logps  # noqa: PLC0415
    except Exception:  # noqa: BLE001  (OLMo-core env not available here)
        return
    t = torch.randn(2, 9)
    lab = torch.randint(-1, 3, (2, 9)).masked_fill(torch.rand(2, 9) < 0.4, -100)
    for avg in (True, False):
        assert torch.allclose(gemma4_dpo.batch_logps(t, lab, avg), _get_batch_logps(t, lab, average_log_prob=avg))


def test_load_persistent_buffers_restores_layer_scalar(tmp_path):
    model = _tiny_model()
    want = [l.layer_scalar.clone() for l in model.model.layers]
    model.save_pretrained(tmp_path)
    with torch.no_grad():
        for l in model.model.layers:
            l.layer_scalar.fill_(1.0)  # what transformers 5.5.3 leaves behind under ZeRO-3 init
    rep = gemma4_dpo.load_persistent_buffers(model, str(tmp_path))
    assert rep["changed"] == len(want) and rep["in_checkpoint"] == len(want)
    assert all(torch.equal(l.layer_scalar, w) for l, w in zip(model.model.layers, want))


def test_recompute_rope_buffers_is_identity_on_plain_load():
    model = _tiny_model()
    x = torch.randint(0, 384, (1, 40))
    with torch.no_grad():
        before = model(input_ids=x).logits
        rep = gemma4_dpo.recompute_rope_buffers(model)
        after = model(input_ids=x).logits
    assert rep and all(v["max_abs_change"] == 0.0 and v["dtype_before"] == "torch.float32" for v in rep.values())
    assert torch.equal(before, after)
    with torch.no_grad():  # a bf16-rounded buffer (what a ZeRO-3 init could leave) is restored to fp32
        model.model.rotary_emb.full_attention_inv_freq.data = model.model.rotary_emb.full_attention_inv_freq.bfloat16()
        rep = gemma4_dpo.recompute_rope_buffers(model)
    assert rep["full_attention_inv_freq"]["dtype_before"] == "torch.bfloat16"
    assert model.model.rotary_emb.full_attention_inv_freq.dtype == torch.float32


# ---------------------------------------------------------------------------------------------- tokenizer / labeler
needs_servable = pytest.mark.skipif(not os.path.isdir(SERVABLE), reason="Gemma-4 servable not on this host")

TOOLS = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                    "required": ["command"],
                },
            },
        }
    ]
)


def _row():
    def call(i, cmd):
        return [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]

    base = [
        {"role": "system", "content": "You are an agent.\n", "reasoning_content": "", "tool_calls": [], "tool_call_ids": []},
        {"role": "user", "content": "List then read.", "reasoning_content": "", "tool_calls": [], "tool_call_ids": []},
    ]

    def side(cmd2):
        return base + [
            {"role": "assistant", "content": "THOUGHT: look.\n\n", "reasoning_content": "secret plan", "tool_calls": call(1, "ls"), "tool_call_ids": ["c1"]},
            {"role": "tool", "content": "a.txt\n\n(exit_code=0)", "reasoning_content": "", "tool_calls": [], "tool_call_ids": ["c1"]},
            {"role": "assistant", "content": "THOUGHT: read.\n\n", "reasoning_content": "more plan", "tool_calls": call(2, cmd2), "tool_call_ids": ["c2"]},
            {"role": "tool", "content": "hello\n\n(exit_code=0)", "reasoning_content": "", "tool_calls": [], "tool_call_ids": ["c2"]},
        ]

    return {"chosen": side("cat a.txt"), "rejected": side("rm a.txt"), "tools": TOOLS, "metadata": {"n_prefix_messages": 2}}


@needs_servable
def test_labeler_segments_and_spans():
    tok = AutoTokenizer.from_pretrained(SERVABLE)
    row = _row()
    conv = gemma4_dpo.normalize_conversation_gemma4(row["chosen"])
    assert conv[3]["tool_call_id"] == "c1" and "reasoning" not in conv[2] and conv[2]["content"] == []
    segs = gemma4_dpo.build_segments(tok, conv, gemma4_dpo.normalize_tools(row["tools"]), 2)
    assert len(segs) == 2  # [prompt + turn 1] and [history + turn 2]
    assert segs[0]["prompts"][0].endswith("<|turn>model\n<|channel>thought\n<channel|>")
    for seg in segs:
        assert "THOUGHT" not in seg["text"] and "plan" not in seg["text"]
        assert "response:bash{" in seg["text"] or seg is segs[0]
    out = gemma4_dpo.preference_gemma4_tail_tokenize_v1(row, tok, max_seq_length=4096)
    for branch, cmd in (("chosen", "cat a.txt"), ("rejected", "rm a.txt")):
        ids, lab, seg = out[f"{branch}_input_ids"], out[f"{branch}_labels"], out[f"{branch}_attention_mask"]
        assert len(ids) == len(lab) == len(seg) and set(seg) == {1, 2}
        spans, cur = [], []
        for t, l in zip(ids, lab):
            if l != -100:
                cur.append(t)
            elif cur:
                spans.append(cur)
                cur = []
        if cur:
            spans.append(cur)
        texts = [tok.decode(s, skip_special_tokens=False) for s in spans]
        assert texts == [
            '<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|><|tool_response>',
            f'<|tool_call>call:bash{{command:<|"|>{cmd}<|"|>}}<tool_call|><|tool_response>',
        ], texts
    small = gemma4_dpo.preference_gemma4_tail_tokenize_v1(row, tok, max_seq_length=50)
    assert all(x == -100 for x in small["chosen_labels"]) and all(x == -100 for x in small["rejected_labels"])


# ------------------------------------------------------------------------------- labeler policy v2 (added 2026-11-08)
def _row_v2(first_content="THOUGHT: look.", n_turns=3):
    """A pair whose turn 1 is a content-bearing tool call; turns 2.. are bare calls (content None, as Gemma-31B sends)."""

    def call(i, cmd):
        return [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]

    base = [{"role": "system", "content": "You are an agent.\n"}, {"role": "user", "content": "Do it."}]

    def side(last):
        msgs = list(base)
        for i in range(1, n_turns + 1):
            content = first_content if i == 1 else None
            cmd = last if i == n_turns else f"echo {i}"
            msgs.append({"role": "assistant", "content": content, "reasoning_content": "hidden", "tool_calls": call(i, cmd)})
            msgs.append({"role": "tool", "content": f"out {i}\n\n(exit_code=0)", "tool_call_ids": [f"c{i}"]})
        return msgs

    return {"chosen": side("cat a.txt"), "rejected": side("rm a.txt"), "tools": TOOLS, "metadata": {"n_prefix_messages": 2}}


def _spans(ids, lab, tok):
    spans, cur = [], []
    for t, lb in zip(ids, lab):
        if lb != -100:
            cur.append(t)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return [tok.decode(s, skip_special_tokens=False) for s in spans]


def test_v2_unlabeled_turns_rule():
    conv = gemma4_dpo.normalize_conversation_gemma4(_row_v2(n_turns=4)["chosen"], keep_call_content=True)
    assert conv[2]["content"] == [{"type": "text", "text": "THOUGHT: look."}] and conv[4]["content"] == []
    assert gemma4_dpo.v2_unlabeled_turns(conv, 2) == {2, 4}  # the content-bearing call and the next assistant turn
    # v1 normalization (default) drops call content, so nothing would be skipped
    conv1 = gemma4_dpo.normalize_conversation_gemma4(_row_v2(n_turns=4)["chosen"])
    assert conv1[2]["content"] == [] and gemma4_dpo.v2_unlabeled_turns(conv1, 2) == set()
    # an empty-string content is not content-bearing; whitespace is (conservative)
    assert not gemma4_dpo.is_content_bearing_call({"tool_calls": [1], "content": [{"type": "text", "text": ""}]})
    assert gemma4_dpo.is_content_bearing_call({"tool_calls": [1], "content": [{"type": "text", "text": "\n"}]})
    assert not gemma4_dpo.is_content_bearing_call({"content": [{"type": "text", "text": "final answer"}]})


@needs_servable
def test_v2_labels_skip_content_turn_and_next_keep_history():
    tok = AutoTokenizer.from_pretrained(SERVABLE)
    row = _row_v2(n_turns=3)
    v2 = gemma4_dpo.preference_gemma4_tail_tokenize_v2(row, tok, max_seq_length=8192)
    v1 = gemma4_dpo.preference_gemma4_tail_tokenize_v1(row, tok, max_seq_length=8192)
    for branch, cmd in (("chosen", "cat a.txt"), ("rejected", "rm a.txt")):
        texts2 = _spans(v2[f"{branch}_input_ids"], v2[f"{branch}_labels"], tok)
        assert texts2 == [f'<|tool_call>call:bash{{command:<|"|>{cmd}<|"|>}}<tool_call|><|tool_response>'], texts2
        full2 = tok.decode(v2[f"{branch}_input_ids"], skip_special_tokens=False)
        assert "THOUGHT: look." in full2 and "hidden" not in full2  # content kept in history, reasoning dropped
        texts1 = _spans(v1[f"{branch}_input_ids"], v1[f"{branch}_labels"], tok)  # v1 unchanged: 3 labeled turns
        assert len(texts1) == 3 and "THOUGHT" not in tok.decode(v1[f"{branch}_input_ids"], skip_special_tokens=False)
    # the labeled turn's context is the exact inference prompt rendered from the history WITH the content
    conv = gemma4_dpo.normalize_conversation_gemma4(row["chosen"], keep_call_content=True)
    prompt, _ = gemma4_dpo.turn_prompt_and_completion(tok, conv, gemma4_dpo.normalize_tools(row["tools"]), 6)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids, lab, seg = v2["chosen_input_ids"], v2["chosen_labels"], v2["chosen_attention_mask"]
    first = next(i for i, x in enumerate(lab) if x != -100)
    start = seg.index(seg[first])
    assert ids[start:first] == p_ids


@needs_servable
def test_v2_equals_v1_without_call_content():
    tok = AutoTokenizer.from_pretrained(SERVABLE)
    row = _row_v2(first_content=None, n_turns=3)
    assert gemma4_dpo.preference_gemma4_tail_tokenize_v2(row, tok, 8192) == gemma4_dpo.preference_gemma4_tail_tokenize_v1(row, tok, 8192)


@needs_servable
def test_v2_side_with_no_labelable_turn_is_dropped_by_filter():
    tok = AutoTokenizer.from_pretrained(SERVABLE)
    row = _row_v2(n_turns=1)  # the only tail turn is content-bearing
    out = gemma4_dpo.preference_gemma4_tail_tokenize_v2(row, tok, 8192)
    assert out["chosen_input_ids"] == [] and out["chosen_labels"] == []
    from open_instruct.dataset_transformation import preference_tulu_filter_v1  # noqa: PLC0415

    assert not preference_tulu_filter_v1(out, tok)


# ------------------------------------------------------------------------- length-grouped sampler (added 2026-11-08)
def _heavy_costs(n=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = (torch.rand(n, 2, generator=g) ** 3 * 30000 + 1000).long().tolist()  # heavy tail like the SafeComp pairs
    return [gemma4_dpo.pair_cost(a, b) for a, b in lens]


@pytest.mark.parametrize("n,P,B,K", [(500, 8, 1, 8), (37, 8, 1, 4), (101, 4, 2, 8), (9, 8, 1, 16)])
def test_length_grouped_order_is_a_seeded_permutation(n, P, B, K):
    costs = _heavy_costs(n)
    o1 = gemma4_dpo.length_grouped_order(costs, P, B, K, seed=42, epoch=0)
    assert sorted(o1) == list(range(n))
    assert o1 == gemma4_dpo.length_grouped_order(costs, P, B, K, seed=42, epoch=0)  # same on every rank
    if n > 2 * P:
        assert o1 != gemma4_dpo.length_grouped_order(costs, P, B, K, seed=42, epoch=1)
        assert o1 != gemma4_dpo.length_grouped_order(costs, P, B, K, seed=1, epoch=0)


def test_group_steps_1_is_plain_random_order():
    import numpy as np  # noqa: PLC0415

    costs = _heavy_costs(64)
    got = gemma4_dpo.length_grouped_order(costs, 8, 1, 1, seed=7, epoch=3)
    assert got == np.random.default_rng([7, 3]).permutation(64).tolist()


def test_length_grouped_balances_micro_steps():
    costs = _heavy_costs(2000)
    rnd = gemma4_dpo.length_grouped_order(costs, 8, 1, 1, seed=42, epoch=0)
    grp = gemma4_dpo.length_grouped_order(costs, 8, 1, 8, seed=42, epoch=0)
    ideal = sum(costs) / 8  # perfect balance lower bound
    w_rnd, w_grp = gemma4_dpo.simulate_wall(rnd, costs, 8), gemma4_dpo.simulate_wall(grp, costs, 8)
    assert w_grp < 0.75 * w_rnd and w_grp >= ideal
    # per-device batch 2: pairs dealt largest-first to the least-loaded rank
    grp2 = gemma4_dpo.length_grouped_order(costs, 4, 2, 8, seed=42, epoch=0)
    rnd2 = gemma4_dpo.length_grouped_order(costs, 4, 2, 1, seed=42, epoch=0)
    # (two pairs per rank already average out part of the imbalance, so the gain is smaller than with B = 1)
    assert gemma4_dpo.simulate_wall(grp2, costs, 4, 2) < 0.9 * gemma4_dpo.simulate_wall(rnd2, costs, 4, 2)


@pytest.mark.parametrize("n", [64, 61])
def test_sampler_through_accelerate_batch_sampler_shard(n):
    """accelerate's BatchSamplerShard (split_batches=False) must give micro-step g of rank r the order's position
    g * P + r: the assumption the sampler's grouping relies on."""
    accel_dl = pytest.importorskip("accelerate.data_loader")
    from torch.utils.data import BatchSampler  # noqa: PLC0415

    P, costs = 8, _heavy_costs(n)
    sampler = gemma4_dpo.LengthGroupedMicrostepSampler(costs, P, 1, 4, seed=5)
    sampler.set_epoch(2)
    order = list(iter(sampler))
    assert order == gemma4_dpo.length_grouped_order(costs, P, 1, 4, 5, 2) and len(sampler) == n
    per_rank = [
        [b[0] for b in accel_dl.BatchSamplerShard(BatchSampler(sampler, 1, False), P, r, split_batches=False)]
        for r in range(P)
    ]
    steps = min(len(x) for x in per_rank)
    for g in range(n // P):
        assert sorted(per_rank[r][g] for r in range(P)) == sorted(order[g * P : (g + 1) * P])
    assert steps == -(-n // P)  # even_batches pads the last micro-step, as with the stock sampler


# -------------------------------------------------------------------------- hybrid attention (added 2026-11-08)
def _hybrid(model):
    gemma4_dpo.register_hybrid_attention()
    model.set_attn_implementation(gemma4_dpo.HYBRID_ATTN)
    return model


def test_reference_varlen_matches_masked_sdpa():
    """Sanity of the CPU stand-in: FA's documented semantics == a dense block-diagonal sliding causal mask."""
    torch.manual_seed(0)
    seg = torch.tensor([1] * 13 + [2] * 21)
    L, Hq, Hkv, D, W = 34, 4, 2, 16, 8
    q, k, v = torch.randn(L, Hq, D), torch.randn(L, Hkv, D), torch.randn(L, Hkv, D)
    cu, mx = gemma4_dpo.cu_seqlens_from_positions(gemma4_dpo.segment_positions(seg)[None])
    assert cu.tolist() == [0, 13, 34] and mx == 21
    got = gemma4_dpo.reference_varlen_attention(q, k, v, cu, cu, mx, mx, softmax_scale=1.0, causal=True, window_size=(W - 1, 0))
    mask = gemma4_dpo.gemma4_masks(seg, W, "sdpa")["sliding_attention"][0, 0]
    kk, vv = k.repeat_interleave(Hq // Hkv, 1), v.repeat_interleave(Hq // Hkv, 1)
    want = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1), attn_mask=mask, scale=1.0
    ).transpose(0, 1)
    assert torch.allclose(got, want, atol=1e-5)


def test_hybrid_masks_and_positions():
    seg = torch.tensor([1] * 5 + [2] * 3)
    m = gemma4_dpo.gemma4_masks(seg, 4, gemma4_dpo.HYBRID_ATTN)
    assert m["sliding_attention"] is None
    assert torch.equal(m["full_attention"], gemma4_dpo.gemma4_masks(seg, 4, "sdpa")["full_attention"])
    with pytest.raises(ValueError):
        gemma4_dpo.cu_seqlens_from_positions(torch.tensor([[1, 2, 0]]))


def test_hybrid_forward_equals_sdpa_values_and_grads(monkeypatch):
    calls = {"n": 0}

    def spy(*a, **kw):
        calls["n"] += 1
        return gemma4_dpo.reference_varlen_attention(*a, **kw)

    monkeypatch.setattr(gemma4_dpo, "_FLASH_VARLEN", spy)
    ref_model, hyb_model = _tiny_model(), _hybrid(_tiny_model())
    assert hyb_model.config._attn_implementation == gemma4_dpo.HYBRID_ATTN
    a, b = torch.randint(0, 384, (1, 13)), torch.randint(0, 384, (1, 21))  # both segments longer than the window (8)
    ids = torch.cat([a, b], 1)
    seg = torch.tensor([1] * 13 + [2] * 21)
    labels = ids.clone()
    labels[:, :4] = -100
    pos = gemma4_dpo.segment_positions(seg)[None]
    ref = ref_model(input_ids=ids, attention_mask=gemma4_dpo.gemma4_masks(seg, 8, "sdpa"), position_ids=pos, dpo_labels=labels)
    got = hyb_model(input_ids=ids, attention_mask=gemma4_dpo.gemma4_masks(seg, 8, gemma4_dpo.HYBRID_ATTN), position_ids=pos,
                    dpo_labels=labels)
    n_sliding = sum(t == "sliding_attention" for t in hyb_model.config.layer_types)
    assert calls["n"] == n_sliding  # sliding layers went to the varlen kernel, global layers did not
    mask = labels[:, 1:] != -100
    assert torch.allclose(got[:, :-1][mask], ref[:, :-1][mask], atol=1e-5, rtol=0)
    ref[:, :-1][mask].sum().backward()
    got[:, :-1][mask].sum().backward()
    for (n, p_ref), (_, p_got) in zip(ref_model.named_parameters(), hyb_model.named_parameters()):
        err = (p_got.grad - p_ref.grad).abs().max()
        assert err <= 1e-5 * p_ref.grad.abs().max() + 1e-7, (n, err)


def test_hybrid_through_separate_forward(monkeypatch):
    monkeypatch.setattr(gemma4_dpo, "_FLASH_VARLEN", gemma4_dpo.reference_varlen_attention)
    ids = torch.randint(0, 384, (1, 30))
    seg = torch.tensor([[1] * 10 + [2] * 16 + [0] * 4])
    labels = ids.clone()
    labels[0, :12] = -100
    labels[0, 26:] = -100
    batch = {"chosen_input_ids": ids, "chosen_attention_mask": seg, "chosen_labels": labels,
             "rejected_input_ids": ids[:, :26], "rejected_attention_mask": seg[:, :26], "rejected_labels": labels[:, :26]}
    with torch.no_grad():
        c0, r0, _ = gemma4_dpo.gemma4_separate_forward(_tiny_model(), batch, average_log_prob=True)
        c1, r1, _ = gemma4_dpo.gemma4_separate_forward(_hybrid(_tiny_model()), batch, average_log_prob=True)
    assert torch.allclose(c0, c1, atol=1e-5) and torch.allclose(r0, r1, atol=1e-5)


# ------------------------------------------------------- chosen-NLL anchor on the DeepSpeed path (added 2026-11-08)
try:
    from open_instruct import dpo_tune_cache_gemma4, dpo_utils  # noqa: E402
except Exception as e:  # noqa: BLE001  (needs the open-instruct training env: deepspeed, accelerate, OLMo-core)
    dpo_tune_cache_gemma4 = None
    _ENTRY_IMPORT_ERROR = repr(e)
needs_entry = pytest.mark.skipif(dpo_tune_cache_gemma4 is None, reason="open-instruct training env not importable")


class _Cache:
    def __init__(self, c, r):
        self.t = {"chosen_logps": c, "rejected_logps": r}

    def __getitem__(self, idx):
        return {k: v[idx.long()] for k, v in self.t.items()}


def _toy(loss_type):
    torch.manual_seed(3)
    B = 4
    args = dpo_utils.DPOConfig(beta=5.0, loss_type=loss_type)
    labels = torch.full((B, 12), -100)
    for b, n in enumerate([3, 5, 8, 11]):
        labels[b, 12 - n :] = 1
    batch = {"index": torch.arange(B), "chosen_labels": labels}
    scale = 1.0 if loss_type == dpo_utils.DPOLossType.dpo_norm else 6.0
    pc = (-(torch.rand(B) * 0.5 + 0.1) * scale).requires_grad_()
    pr = (-(torch.rand(B) * 0.5 + 0.2) * scale).requires_grad_()
    cache = _Cache(-(torch.rand(B) * 0.5 + 0.1) * scale, -(torch.rand(B) * 0.5 + 0.1) * scale)
    return args, batch, pc, pr, cache


@needs_entry
@pytest.mark.parametrize("loss_type", ["dpo_norm", "dpo"])
@pytest.mark.parametrize("coef", [1.0, 0.2])
def test_chosen_nll_wrapper_equals_olmo_core_definition(loss_type, coef):
    lt = dpo_utils.DPOLossType(loss_type)
    args, batch, pc, pr, cache = _toy(lt)
    wrapped = dpo_tune_cache_gemma4.make_compute_loss_with_chosen_nll(coef)
    got = wrapped(args, batch, pc, pr, cache)[0].mean()  # what dpo_tune_cache does: loss = losses.mean()
    # the OLMo-core micro-batch loss (olmo_core_train_modules._compute_microbatch_loss, same calls in the same order)
    pc2, pr2 = pc.detach().clone().requires_grad_(), pr.detach().clone().requires_grad_()
    losses, _, _ = dpo_utils.compute_loss(args, batch, pc2, pr2, cache)
    want, nll = dpo_utils.add_chosen_nll(dpo_utils.apply_dpo_loss_coef(losses.mean(), 1.0), pc2, batch, coef,
                                         lt.is_average_loss)
    assert torch.allclose(got, want, atol=1e-6, rtol=0)
    got.backward()
    want.backward()
    assert torch.allclose(pc.grad, pc2.grad, atol=1e-6) and torch.allclose(pr.grad, pr2.grad, atol=1e-6)
    if lt == dpo_utils.DPOLossType.dpo_norm:  # hand-computed: -log s(b*((pc-pr)-(rc-rr))) - coef*pc, mean over pairs
        rc, rr = cache.t["chosen_logps"], cache.t["rejected_logps"]
        m = (pc.detach() - pr.detach()) - (rc - rr)
        hand = (-torch.nn.functional.logsigmoid(5.0 * m) - coef * pc.detach()).mean()
        assert torch.allclose(got.detach(), hand, atol=1e-6)
        # dL/dpc_i = (-b * s(-b m_i) - coef) / B
        assert torch.allclose(pc.grad, (-5.0 * torch.sigmoid(-5.0 * m) - coef) / 4, atol=1e-6)


@needs_entry
def test_apply_run_options_defaults_install_nothing(monkeypatch):
    import types  # noqa: PLC0415

    monkeypatch.setattr(dpo_utils, "compute_loss", dpo_tune_cache_gemma4._ORIG_COMPUTE_LOSS)
    monkeypatch.setattr(dpo_tune_cache_gemma4.dpo_tune_cache, "RandomSampler", dpo_tune_cache_gemma4._ORIG_RANDOM_SAMPLER)
    monkeypatch.setattr(dpo_utils, "REFERENCE_LOGPROBS_CACHE_PATH", "/x/cache")
    monkeypatch.setitem(dpo_tune_cache_gemma4._RUN, "chosen_nll_coef", 0.0)
    monkeypatch.setitem(dpo_tune_cache_gemma4._RUN, "hybrid_attention", False)
    args = types.SimpleNamespace(loss_type=dpo_utils.DPOLossType.dpo_norm, chosen_nll_coef=0.0,
                                 gemma4_length_grouped_batches=False, gemma4_hybrid_flash_attention=False,
                                 per_device_train_batch_size=1, gemma4_group_steps=8, seed=42, gemma4_cost_l0=58350.0)
    dpo_tune_cache_gemma4.apply_run_options(args)
    assert dpo_utils.compute_loss is dpo_tune_cache_gemma4._ORIG_COMPUTE_LOSS
    assert dpo_tune_cache_gemma4.dpo_tune_cache.RandomSampler is dpo_tune_cache_gemma4._ORIG_RANDOM_SAMPLER
    assert dpo_utils.REFERENCE_LOGPROBS_CACHE_PATH == "/x/cache" and not dpo_tune_cache_gemma4._RUN["hybrid_attention"]
    # all three on
    args = types.SimpleNamespace(**{**vars(args), "chosen_nll_coef": 1.0, "gemma4_length_grouped_batches": True,
                                    "gemma4_hybrid_flash_attention": True})
    dpo_tune_cache_gemma4.apply_run_options(args)
    assert dpo_utils.compute_loss._gemma4_chosen_nll_coef == 1.0 and args.chosen_nll_coef == 0.0
    assert dpo_tune_cache_gemma4._RUN["chosen_nll_coef"] == 1.0
    assert dpo_utils.REFERENCE_LOGPROBS_CACHE_PATH == "/x/cache/gemma4_hybrid"
    ds = {"chosen_input_ids": [[1] * 10, [1] * 3000], "rejected_input_ids": [[1] * 20, [1] * 10]}
    smp = dpo_tune_cache_gemma4.dpo_tune_cache.RandomSampler(ds, generator=torch.Generator().manual_seed(42))
    assert isinstance(smp, gemma4_dpo.LengthGroupedMicrostepSampler) and len(smp) == 2 and smp.costs[1] > smp.costs[0]


@needs_entry
def test_cli_flags_parse_with_stock_defaults():
    from open_instruct.dataset_transformation import TokenizerConfig  # noqa: PLC0415
    from open_instruct.utils import ArgumentParserPlus  # noqa: PLC0415

    p = ArgumentParserPlus((dpo_tune_cache_gemma4.Gemma4DPOExperimentConfig, TokenizerConfig))
    base = ["--model_name_or_path", "m", "--mixer_list", "x.jsonl", "1.0"]
    a, _ = p.parse_args_into_dataclasses(args=base)
    assert a.chosen_nll_coef == 0.0 and not a.gemma4_length_grouped_batches and not a.gemma4_hybrid_flash_attention
    assert a.gemma4_group_steps == 8
    a, _ = p.parse_args_into_dataclasses(args=base + ["--chosen_nll_coef", "1.0", "--gemma4_length_grouped_batches",
                                                      "--gemma4_hybrid_flash_attention", "--gemma4_group_steps", "4"])
    assert a.chosen_nll_coef == 1.0 and a.gemma4_length_grouped_batches and a.gemma4_hybrid_flash_attention
    assert a.gemma4_group_steps == 4


# --- --dpo_loss_coef on the Gemma-4 path (2026-11-12, Gemma SFT-on-chosen control) -----------------------------------


@needs_entry
@pytest.mark.parametrize("loss_type", ["dpo_norm", "dpo"])
def test_dpo_loss_coef_zero_wrapper_equals_olmo_core_sftc(loss_type):
    """0 * DPO + 1.0 * NLL(chosen) (the Qwen RS-sftc loss) equals the OLMo-core micro-batch loss in value and gradient;
    the rejected side gets exactly zero gradient."""
    lt = dpo_utils.DPOLossType(loss_type)
    args, batch, pc, pr, cache = _toy(lt)
    wrapped = dpo_tune_cache_gemma4.make_compute_loss_with_chosen_nll(1.0, 0.0)
    assert wrapped._gemma4_dpo_loss_coef == 0.0 and wrapped._gemma4_chosen_nll_coef == 1.0
    out, chosen_rewards, rejected_rewards = wrapped(args, batch, pc, pr, cache)
    got = out.mean()
    pc2, pr2 = pc.detach().clone().requires_grad_(), pr.detach().clone().requires_grad_()
    losses, cr2, rr2 = dpo_utils.compute_loss(args, batch, pc2, pr2, cache)
    want, _ = dpo_utils.add_chosen_nll(dpo_utils.apply_dpo_loss_coef(losses.mean(), 0.0), pc2, batch, 1.0,
                                       lt.is_average_loss)
    assert torch.allclose(got, want, atol=1e-6, rtol=0)
    # rewards are still computed (logged as in the DPO run)
    assert torch.equal(chosen_rewards, cr2) and torch.equal(rejected_rewards, rr2)
    got.backward()
    want.backward()
    assert torch.allclose(pc.grad, pc2.grad, atol=1e-6)
    assert torch.equal(pr.grad, torch.zeros_like(pr.grad)) and torch.equal(pr2.grad, torch.zeros_like(pr2.grad))
    if lt == dpo_utils.DPOLossType.dpo_norm:  # pure NLL: L = -mean(pc), dL/dpc_i = -1/B
        assert torch.allclose(got.detach(), -pc.detach().mean(), atol=1e-7)
        assert torch.allclose(pc.grad, torch.full_like(pc.grad, -0.25), atol=1e-7)


@needs_entry
@pytest.mark.parametrize("loss_type", ["dpo_norm", "dpo"])
def test_dpo_loss_coef_default_reproduces_previous_wrapper(loss_type):
    """dpo_loss_coef 1.0 (default) computes exactly the pre-2026-11-12 wrapper: losses + coef * NLL (bitwise)."""
    lt = dpo_utils.DPOLossType(loss_type)
    args, batch, pc, pr, cache = _toy(lt)
    got = dpo_tune_cache_gemma4.make_compute_loss_with_chosen_nll(1.0)(args, batch, pc, pr, cache)[0]
    losses, _, _ = dpo_tune_cache_gemma4._ORIG_COMPUTE_LOSS(args, batch, pc, pr, cache)
    old = losses + 1.0 * dpo_utils.chosen_nll_loss(pc, batch.get("chosen_labels"), lt.is_average_loss)
    assert torch.equal(got, old)
    with pytest.raises(ValueError):
        dpo_tune_cache_gemma4.make_compute_loss_with_chosen_nll(1.0, -0.5)


@needs_entry
def test_apply_run_options_dpo_loss_coef_zero(monkeypatch):
    import types  # noqa: PLC0415

    monkeypatch.setattr(dpo_utils, "compute_loss", dpo_tune_cache_gemma4._ORIG_COMPUTE_LOSS)
    monkeypatch.setattr(dpo_tune_cache_gemma4.dpo_tune_cache, "RandomSampler", dpo_tune_cache_gemma4._ORIG_RANDOM_SAMPLER)
    monkeypatch.setitem(dpo_tune_cache_gemma4._RUN, "chosen_nll_coef", 0.0)
    monkeypatch.setitem(dpo_tune_cache_gemma4._RUN, "dpo_loss_coef", 1.0)
    args = types.SimpleNamespace(loss_type=dpo_utils.DPOLossType.dpo_norm, chosen_nll_coef=1.0, dpo_loss_coef=0.0,
                                 gemma4_length_grouped_batches=False, gemma4_hybrid_flash_attention=False,
                                 per_device_train_batch_size=1, gemma4_group_steps=8, seed=42, gemma4_cost_l0=58350.0)
    dpo_tune_cache_gemma4.apply_run_options(args)
    assert dpo_utils.compute_loss._gemma4_chosen_nll_coef == 1.0 and dpo_utils.compute_loss._gemma4_dpo_loss_coef == 0.0
    assert args.chosen_nll_coef == 0.0 and args.dpo_loss_coef == 1.0  # what dpo_tune_cache.main accepts
    assert dpo_tune_cache_gemma4._RUN["chosen_nll_coef"] == 1.0 and dpo_tune_cache_gemma4._RUN["dpo_loss_coef"] == 0.0
    # the DPO run's options leave dpo_loss_coef at 1.0
    monkeypatch.setattr(dpo_utils, "compute_loss", dpo_tune_cache_gemma4._ORIG_COMPUTE_LOSS)
    args = types.SimpleNamespace(**{**vars(args), "chosen_nll_coef": 1.0, "dpo_loss_coef": 1.0})
    dpo_tune_cache_gemma4.apply_run_options(args)
    assert dpo_utils.compute_loss._gemma4_dpo_loss_coef == 1.0 and dpo_tune_cache_gemma4._RUN["dpo_loss_coef"] == 1.0


@needs_entry
def test_cli_dpo_loss_coef_parse_and_check_args():
    from open_instruct.dataset_transformation import TokenizerConfig  # noqa: PLC0415
    from open_instruct.utils import ArgumentParserPlus  # noqa: PLC0415

    p = ArgumentParserPlus((dpo_tune_cache_gemma4.Gemma4DPOExperimentConfig, TokenizerConfig))
    base = ["--model_name_or_path", "m", "--mixer_list", "x.jsonl", "1.0", "--transform_fn",
            "preference_gemma4_tail_tokenize_v2", "preference_tulu_filter_v1", "--concatenated_forward", "false",
            "--packing", "false", "--attn_implementation", "torch", "--loss_type", "dpo_norm"]
    a, _ = p.parse_args_into_dataclasses(args=base)
    assert a.dpo_loss_coef == 1.0
    a, _ = p.parse_args_into_dataclasses(args=base + ["--chosen_nll_coef", "1.0", "--dpo_loss_coef", "0"])
    assert a.dpo_loss_coef == 0.0 and a.chosen_nll_coef == 1.0
    dpo_tune_cache_gemma4.check_args(a)  # accepted
    a, _ = p.parse_args_into_dataclasses(args=base + ["--dpo_loss_coef", "0"])
    with pytest.raises(ValueError, match="needs --chosen_nll_coef"):
        dpo_tune_cache_gemma4.check_args(a)
