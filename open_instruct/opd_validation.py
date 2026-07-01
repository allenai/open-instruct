from typing import Any

import transformers

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def validate_pure_opd_reference_config(*, opd_use_task_rewards: bool, beta: float, load_ref_policy: bool) -> None:
    if opd_use_task_rewards:
        return
    if beta == 0.0 and not load_ref_policy:
        return
    raise ValueError(
        "Pure OPD (`opd_use_task_rewards=False`) requires `--beta 0.0` and "
        "`--load_ref_policy false` because the GRPO task loss and reference KL are not used. "
        f"Got beta={beta} and load_ref_policy={load_ref_policy}."
    )


def validate_teacher_student_tokenizer_compatibility(
    *, student_tokenizer: Any, teacher_tokenizer: Any, student_name: str, teacher_name: str
) -> None:
    """Validate that teacher token ids can be interpreted by the student.

    OPD stores teacher top-k token ids and gathers student logprobs at those ids.
    That is only meaningful when every teacher vocab token has the same id in
    the student tokenizer. Student-only tokens above the teacher vocab range are
    allowed because Open Instruct may add a pad token that the teacher will never
    emit as a model top-k id.
    """
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    teacher_vocab_size = max(teacher_vocab.values(), default=-1) + 1

    missing_tokens = []
    mismatched_tokens = []
    for token, teacher_id in teacher_vocab.items():
        student_id = student_vocab.get(token)
        if student_id is None:
            missing_tokens.append((token, teacher_id))
        elif student_id != teacher_id:
            mismatched_tokens.append((token, teacher_id, student_id))

    conflicting_student_tokens = [
        (token, student_id)
        for token, student_id in student_vocab.items()
        if student_id < teacher_vocab_size and teacher_vocab.get(token) != student_id
    ]

    special_id_mismatches = []
    for attr in ("bos_token_id", "eos_token_id", "unk_token_id"):
        teacher_value = getattr(teacher_tokenizer, attr, None)
        student_value = getattr(student_tokenizer, attr, None)
        if teacher_value != student_value:
            special_id_mismatches.append((attr, teacher_value, student_value))

    if missing_tokens or mismatched_tokens or conflicting_student_tokens or special_id_mismatches:
        details = []
        if missing_tokens:
            details.append(f"missing teacher tokens in student vocab: {missing_tokens[:5]}")
        if mismatched_tokens:
            details.append(f"token id mismatches: {mismatched_tokens[:5]}")
        if conflicting_student_tokens:
            details.append(f"student tokens conflict inside teacher vocab range: {conflicting_student_tokens[:5]}")
        if special_id_mismatches:
            details.append(f"special token id mismatches: {special_id_mismatches[:5]}")
        raise ValueError(
            "OPD requires teacher and student tokenizers to share token ids because teacher top-k ids are "
            "used to gather student logprobs. "
            f"Student tokenizer: {student_name}; teacher tokenizer: {teacher_name}. " + " ".join(details)
        )


def validate_teacher_student_output_vocab(
    *,
    student_model_name_or_path: str,
    student_revision: str | None,
    student_vocab_size: int | None = None,
    teacher_model_name_or_path: str,
    teacher_revision: str | None,
    trust_remote_code: bool,
) -> None:
    """Validate that the teacher's output vocab fits within the student's.

    The tokenizer check guarantees teacher and student agree on token *ids*. This
    additionally guarantees the teacher's model *output dimension* is not wider
    than the student's: teacher top-k ids are used as gather indices into the
    student model's logits, so a teacher whose vocab exceeds the student's could
    emit an id that the learner cannot score. The student dim is read from the
    already-built learner config when available, otherwise from its HF config. If
    a non-HF config cannot be loaded, the learner forward still hard-errors on
    out-of-range teacher ids.
    """

    def _vocab_size(name: str, revision: str | None, role: str) -> int | None:
        try:
            config = transformers.AutoConfig.from_pretrained(
                name, revision=revision, trust_remote_code=trust_remote_code
            )
        except Exception as e:  # noqa: BLE001 - config may be an OLMo-core checkpoint or otherwise unloadable
            logger.warning("Could not load %s config (%s) to validate OPD output vocab size; skipping check.", role, e)
            return None
        return getattr(config, "vocab_size", None)

    teacher_output_vocab_size = _vocab_size(teacher_model_name_or_path, teacher_revision, "teacher")
    student_output_vocab_size = (
        student_vocab_size
        if student_vocab_size is not None
        else _vocab_size(student_model_name_or_path, student_revision, "student")
    )
    if teacher_output_vocab_size is None or student_output_vocab_size is None:
        return
    if teacher_output_vocab_size > student_output_vocab_size:
        raise ValueError(
            f"Teacher output vocab ({teacher_output_vocab_size}) is larger than "
            f"student output vocab ({student_output_vocab_size}). "
            "Teacher top-k token ids could exceed the student's logit range and fail during learner gather. "
            f"Student: {student_model_name_or_path}; teacher: {teacher_model_name_or_path}."
        )
