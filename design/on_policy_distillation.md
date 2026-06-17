# Design Doc: On-Policy Distillation (OPD) for open-instruct

Status: **Draft / proposal**
Scope: Add on-policy distillation and on-policy self-distillation to the GRPO RL
infrastructure, behind a single config-driven teacher abstraction.

---

## 1. Goal

Add the ability to train a **student** policy by having it generate rollouts
**on its own policy** (as GRPO already does) and then training it to match a
**teacher**'s per-token distribution over those rollouts.

A single framework should cover both cases, selected purely by config:

- **On-policy distillation (OPD)**: the teacher is a *different* (typically
  larger) frozen model — e.g. a 4B student distilling from a 32B teacher.
- **On-policy self-distillation**: the teacher is the *student itself* — a
  frozen copy of the starting checkpoint, or a slow EMA of the student.

The teacher is "just a parameter." If `--teacher_model_name_or_path` is unset it
defaults to the student's initial checkpoint (self-distillation); if set, it
points at any other model (OPD). In both cases the teacher is hosted the same
way and scored the same way.

### Why on-policy (vs. the off-policy distillation we already have)

The repo already has an *off-policy* distillation data pipeline
(`sample_logits_vllm.py` + `distillkit/compression/`): run a teacher over a
**fixed** dataset, extract top-k logprobs, compress, store. Nothing trains on
those logprobs yet, but more importantly the tokens are static.

On-policy distillation is fundamentally different: the tokens are the student's
**fresh rollouts**, which change every step as the student learns. Teacher
logprobs therefore **cannot be precomputed** — the teacher must score live
samples inside the training loop. This eliminates the train/inference
distribution mismatch ("exposure bias") of SFT-on-teacher-data, gives a **dense
per-token signal** (vs. sparse RL reward), and is far more sample-efficient than
reward-based RL.

---

## 2. Key insight: GRPO already implements ~80% of OPD

The conceptual mapping is: **OPD ≈ GRPO where the dense per-token "reward" is how
much the teacher likes each token the student just produced.** The existing loss
already supports this shape.

`compute_grpo_loss` (`open_instruct/grpo_utils.py:504`):

```python
if ref_logprobs is not None:
    ref_logprobs_diff = (new_logprobs - ref_logprobs).clamp(-40.0, 40.0)
    kl_all = model_utils.estimate_kl(ref_logprobs_diff, ratio)
    kl = kl_all[config.kl_estimator]
else:
    kl = torch.zeros_like(pg_loss)
return pg_loss, clipfrac, kl
```

and in `PolicyTrainerRayProcess.step` (`grpo_fast.py:718`):

```python
per_token_loss_BT = pg_loss_BT + self.args.beta * kl_BT
```

`estimate_kl` (`model_utils.py:793`) estimator `[0]` is the linear
`new_logprobs - ref_logprobs` — i.e. the exact **sampled-token estimator of
`KL(student ‖ ref)`** over the student's own tokens. So if we:

- set `ref_logprobs := teacher_logprobs`,
- set `advantages := 0` (so `pg_loss = 0`), and
- set `beta := 1`,

then `per_token_loss = log π_student(x_t) − log π_teacher(x_t)` summed over the
student's sampled tokens — **this is exactly the on-policy distillation loss.**

Two more existing facts make the framework natural:

1. **Rollouts are already on-policy.** vLLM generation produces
   `GenerationResult` carrying `responses`, `masks`, and per-token sampler
   `logprobs`, which flow through `pack_sequences` (`rl_utils.py:227`) into
   `PackedSequences.vllm_logprobs` and on to the loss. Teacher logprobs can ride
   the exact same path as a sibling field.
2. **A "moving reference" already exists.** `update_ref_policy`
   (`grpo_fast.py:568`) does Polyak EMA `ref ← (1−α)·ref + α·student`, gated by
   `ref_policy_update_freq`. This is the self-distillation "slow teacher"
   primitive.

### What is actually missing

1. **Teacher ≠ student size.** `load_ref_policy` (`model_utils.py:271`) loads
   the *policy's own path* onto the trainer GPUs, sharing the DeepSpeed `mpu`.
   Fine for self-distillation; infeasible for a large external teacher.
2. **Teacher hosting that scales.** We want the teacher on its own vLLM pool
   (decision: Option B), scoring student tokens via teacher-forcing.
3. **Objective shape.** Today KL is a small *penalty* on top of an RL reward.
   OPD wants KL-to-teacher to be the *primary* objective (advantage term
   zeroed/blended), defaulting to the reverse-KL estimator.
4. **Plumbing.** Per-token teacher logprobs aligned to the student's exact
   rollout tokens must be carried from generation → packing → collation → loss.

---

## 3. Architecture (unified, config-driven)

```
                         ┌─────────────────────────────────────────┐
                         │             Ray cluster                  │
                         │                                          │
  prompts ──▶ DataPreparationActor                                  │
                │  1. dispatch prompts                              │
                ▼                                                   │
        ┌──────────────────┐   student rollouts (tokens,           │
        │ STUDENT vLLM pool │── per-token sampler logprobs) ──┐     │
        └──────────────────┘                                  │     │
                ▲   ▲                                         ▼     │
   weight sync  │   │                          ┌───────────────────┐│
   (every step) │   │   2. teacher-force the   │ TEACHER vLLM pool ││
                │   │      student's exact      │  (frozen, or EMA  ││
                │   │      query+response  ────▶│   for self-distill)││
                │   │      via prompt_logprobs  └───────────────────┘│
                │   │                                  │             │
                │   │   3. per-token teacher logprobs  │             │
                │   └──────────────────────────────────┘             │
                ▼                                                    │
        pack_sequences  ──▶  CollatedBatchData                       │
        (student logprobs + teacher logprobs + advantages)          │
                │                                                    │
                ▼                                                    │
        ┌──────────────────────┐                                    │
        │ PolicyTrainerRayProcess│  loss = α·pg_loss + β·KL(student‖teacher)
        │   (DeepSpeed learner)  │  backprop through student         │
        └──────────────────────┘                                    │
                         └─────────────────────────────────────────┘
```

### The teacher abstraction

A teacher is fully described by:

| Field | Meaning |
|---|---|
| `teacher_model_name_or_path` | Model to load on the teacher vLLM pool. **Default: student's init checkpoint** ⇒ self-distillation. |
| `teacher_update_mode` | `frozen` (load once, never sync) \| `ema` (push EMA of student every N steps) \| `online` (push student weights every step). |
| `teacher_num_engines`, `teacher_tensor_parallel_size` | Resource plan for the teacher pool (independent of student). |
| `teacher_ema_alpha`, `teacher_update_freq` | Only for `ema`/`online`. |

Self-distillation = `teacher_model_name_or_path` unset (or == student) +
`teacher_update_mode ∈ {frozen, ema}`. OPD from a big teacher =
`teacher_model_name_or_path` set + `teacher_update_mode=frozen`. Same code path
either way.

### Teacher hosting & scoring (Option B — vLLM pool)

- Stand up a **second pool of `LLMRayActor`s** for the teacher, created by a
  variant of `create_vllm_engines` (`vllm_utils.py:1188`) with its own
  placement group and TP size.
- After student rollouts return, the `DataPreparationActor` sends each
  `query + response` token sequence to the teacher pool requesting
  **`prompt_logprobs`** (teacher-forcing — no generation, `max_tokens` small),
  and extracts per-token teacher logprobs. **This reuses the exact extraction
  logic already in `sample_logits_vllm.py:82-129`** (`process_prompt_logprobs`,
  `FlatLogprobs` handling). For the sampled-token estimator we only need the
  teacher logprob *of each student token*, which is a thin slice of
  `prompt_logprobs`.
- **Weight management by mode:**
  - `frozen`: load teacher weights once; never broadcast. (External teacher, or
    self-distill from a fixed checkpoint.)
  - `ema`/`online`: reuse `broadcast_weights_to_vllm` (`vllm_utils.py:1383`) and
    the EMA math in `update_ref_policy` to push (EMA-)student weights to the
    teacher pool every `teacher_update_freq` steps. No optimizer state, so this
    is cheaper than the student sync.

This is the only design that scales to "small student, huge frozen teacher,"
and it maximally reuses existing code (vLLM actor pool, weight broadcast,
prompt-logprob extraction).

---

## 4. Loss

Default OPD objective (per token `t` over the student's response tokens):

```
L_distill(t) = D_KL( π_student(·|s_t) ‖ π_teacher(·|s_t) )
```

Estimators (reuse `estimate_kl`):
- **Sampled-token reverse KL** (estimator `[0]`): `log π_student(x_t) − log
  π_teacher(x_t)`. Needs only the teacher logprob of the sampled token ⇒ cheap,
  exactly what `prompt_logprobs` gives. **Default.**
- **k3 estimator** (estimator `[2]`, the repo's preferred low-variance form):
  worth A/B-ing.
- **Exact top-k KL** (future): use the teacher's full top-k distribution (the
  compression codec already extracts top-k) for a lower-variance, full-
  distribution KL. Larger payload; deferred.

Blending knob, so we can interpolate RL ↔ pure distillation:

```
per_token_loss = w_rl · pg_loss + w_distill · KL(student ‖ teacher)
```

- Pure OPD: `w_rl = 0`, `w_distill = 1` (advantages effectively unused).
- Distillation-regularized RL: both nonzero (this is essentially today's
  behavior with `beta`, generalized).

We keep `ρ`-correction and clipping available but they are no-ops when
`w_rl = 0`.

---

## 5. Data flow / plumbing changes

Teacher logprobs travel parallel to the existing `vllm_logprobs`:

1. **`data_types.py`**: add `teacher_logprobs` to the rollout/`GenerationResult`
   and `PackedSequences` / `CollatedBatchData` structures.
2. **`data_loader.py`** (`DataPreparationActor._data_preparation_loop`): after
   `accumulate_inference_batches` returns student rollouts, issue the teacher
   teacher-forcing scoring round-trip and attach `teacher_logprobs` aligned to
   each response's tokens.
3. **`rl_utils.py`** (`pack_sequences`): pack `teacher_logprobs` alongside
   `vllm_logprobs` (same masking/alignment, offset-by-one for next-token).
4. **`grpo_fast.py`** (`PolicyTrainerRayProcess.step`): source
   `ref_logprobs := teacher_logprobs` from the batch (instead of / in addition
   to a trainer-side ref forward pass) and feed `compute_grpo_loss` with the
   chosen `w_rl`/`w_distill`.

---

## 6. Config / args (in `grpo_fast.py` Args)

New fields (names provisional):

```
# --- teacher / distillation ---
distillation_enabled: bool = False
teacher_model_name_or_path: str | None = None   # None ⇒ self-distill from init ckpt
teacher_update_mode: Literal["frozen", "ema", "online"] = "frozen"
teacher_ema_alpha: float = 0.0
teacher_update_freq: int | None = None
teacher_num_engines: int = 1
teacher_tensor_parallel_size: int = 1
distill_kl_estimator: int = 0          # index into estimate_kl
w_distill: float = 1.0
w_rl: float = 0.0                      # 0 ⇒ pure distillation
```

Validation: if `distillation_enabled` and `teacher_model_name_or_path is None`,
set teacher = policy init checkpoint and require `teacher_update_mode != online`
unless explicitly intended.

---

## 7. Phased implementation plan

### Phase 0 — Self-distillation via the existing reference path (smallest viable)
- Reuse `load_ref_policy` + `update_ref_policy` (trainer-side, same-size
  teacher) to validate the **objective and metrics** end-to-end with near-zero
  new infra.
- Add `w_rl`/`w_distill`/`distill_kl_estimator`; set `advantages → 0` path.
- Deliverable: self-distillation run on `single_gpu_on_beaker.sh`-style script;
  confirm KL decreases and student tracks teacher.

### Phase 1 — Teacher vLLM pool + teacher-forcing scoring (the real framework)
- Add teacher engine pool creation + `score_logprobs` (teacher-forcing) in
  `vllm_utils.py`, reusing `sample_logits_vllm` extraction.
- Plumb `teacher_logprobs` through `data_types`/`data_loader`/`rl_utils`.
- Switch the loss to consume batch-carried `teacher_logprobs`.
- `frozen` teacher first (covers external teacher + frozen self-distill).

### Phase 2 — EMA/online teacher weight sync + exact top-k KL (optional)
- Push (EMA-)student weights to teacher pool via `broadcast_weights_to_vllm`.
- Optional full-distribution top-k KL using the compression codec's extraction.

---

## 8. File-level touch points

| File | Change |
|---|---|
| `open_instruct/grpo_fast.py` | New Args; teacher pool wiring; `step()` sources teacher logprobs; objective blend. |
| `open_instruct/grpo_utils.py` | Generalize/branch `compute_grpo_loss` for `w_rl`/`w_distill`; distill metrics. |
| `open_instruct/model_utils.py` | (Phase 0) allow `load_ref_policy` to take a distinct teacher path. |
| `open_instruct/vllm_utils.py` | Teacher engine pool; `score_logprobs` teacher-forcing entrypoint; (Phase 2) teacher weight sync. |
| `open_instruct/sample_logits_vllm.py` | Factor out `process_prompt_logprobs` for reuse by the scorer. |
| `open_instruct/data_loader.py` | Teacher scoring round-trip in `DataPreparationActor`. |
| `open_instruct/rl_utils.py`, `data_types.py` | Carry `teacher_logprobs` through packing/collation. |
| `scripts/train/debug/` | New `opd_*.sh` launchers (self-distill + external-teacher). |
| `CHANGELOG.md` | Entry with PR link (on PR). |
| tests | `test_grpo_utils.py` (loss), `test_vllm_utils*` (scoring), data-loader plumbing. |

---

## 9. Open questions / risks

- **Tokenizer compatibility** for external teachers: OPD requires the teacher
  and student to share a tokenizer/vocab so per-token logprobs align. Need a
  validation check; cross-tokenizer distillation is out of scope.
- **Teacher throughput**: teacher-forcing the full `query+response` every step
  adds an inference round-trip. Sizing the teacher pool vs. student pool is a
  resource-plan question (cf. `grpo_fast_resource_plan.py`).
- **Estimator variance**: sampled-token vs. k3 vs. exact top-k — empirical.
- **`online` self-distillation** (teacher == current student, no lag) is
  degenerate (KL → 0); we should default self-distillation to `frozen` or
  `ema`.
- **Numerical**: teacher logprobs from vLLM vs. a trainer forward pass can
  differ slightly (kernels/precision); the existing `ρ`-correction machinery is
  the precedent for handling train/infer mismatch.
