import unittest

from open_instruct import opd_validation


class FakeTokenizer:
    def __init__(
        self,
        vocab: dict[str, int],
        *,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        unk_token_id: int | None = None,
    ):
        self._vocab = vocab
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class TestPureOPDReferenceConfig(unittest.TestCase):
    def test_allows_task_reward_mode(self):
        opd_validation.validate_pure_opd_reference_config(opd_use_task_rewards=True, beta=0.05, load_ref_policy=True)

    def test_allows_pure_opd_without_reference_settings(self):
        opd_validation.validate_pure_opd_reference_config(opd_use_task_rewards=False, beta=0.0, load_ref_policy=False)

    def test_rejects_pure_opd_with_beta(self):
        with self.assertRaisesRegex(ValueError, "requires `--beta 0.0`"):
            opd_validation.validate_pure_opd_reference_config(
                opd_use_task_rewards=False, beta=0.1, load_ref_policy=False
            )

    def test_rejects_pure_opd_with_ref_policy(self):
        with self.assertRaisesRegex(ValueError, "requires `--beta 0.0`"):
            opd_validation.validate_pure_opd_reference_config(
                opd_use_task_rewards=False, beta=0.0, load_ref_policy=True
            )


class TestTokenizerCompatibility(unittest.TestCase):
    def test_allows_identical_vocab_and_special_ids(self):
        student = FakeTokenizer({"a": 0, "b": 1, "<eos>": 2}, eos_token_id=2)
        teacher = FakeTokenizer({"a": 0, "b": 1, "<eos>": 2}, eos_token_id=2)

        opd_validation.validate_teacher_student_tokenizer_compatibility(
            student_tokenizer=student, teacher_tokenizer=teacher, student_name="student", teacher_name="teacher"
        )

    def test_allows_student_extra_token_above_teacher_vocab_range(self):
        student = FakeTokenizer({"a": 0, "b": 1, "<eos>": 2, "<pad>": 3}, eos_token_id=2)
        teacher = FakeTokenizer({"a": 0, "b": 1, "<eos>": 2}, eos_token_id=2)

        opd_validation.validate_teacher_student_tokenizer_compatibility(
            student_tokenizer=student, teacher_tokenizer=teacher, student_name="student", teacher_name="teacher"
        )

    def test_rejects_missing_teacher_token(self):
        student = FakeTokenizer({"a": 0, "<eos>": 2}, eos_token_id=2)
        teacher = FakeTokenizer({"a": 0, "b": 1, "<eos>": 2}, eos_token_id=2)

        with self.assertRaisesRegex(ValueError, "missing teacher tokens"):
            opd_validation.validate_teacher_student_tokenizer_compatibility(
                student_tokenizer=student, teacher_tokenizer=teacher, student_name="student", teacher_name="teacher"
            )

    def test_rejects_mismatched_token_id(self):
        student = FakeTokenizer({"a": 0, "b": 2, "<eos>": 1}, eos_token_id=1)
        teacher = FakeTokenizer({"a": 0, "b": 1, "<eos>": 2}, eos_token_id=2)

        with self.assertRaisesRegex(ValueError, "token id mismatches"):
            opd_validation.validate_teacher_student_tokenizer_compatibility(
                student_tokenizer=student, teacher_tokenizer=teacher, student_name="student", teacher_name="teacher"
            )

    def test_rejects_student_token_conflict_inside_teacher_vocab_range(self):
        student = FakeTokenizer({"a": 0, "b": 1, "<eos>": 2, "student-only": 1}, eos_token_id=2)
        teacher = FakeTokenizer({"a": 0, "b": 1, "<eos>": 2}, eos_token_id=2)

        with self.assertRaisesRegex(ValueError, "student tokens conflict"):
            opd_validation.validate_teacher_student_tokenizer_compatibility(
                student_tokenizer=student, teacher_tokenizer=teacher, student_name="student", teacher_name="teacher"
            )

    def test_rejects_special_token_mismatch(self):
        student = FakeTokenizer({"a": 0, "<eos>": 1}, eos_token_id=1)
        teacher = FakeTokenizer({"a": 0, "<eos>": 1}, eos_token_id=0)

        with self.assertRaisesRegex(ValueError, "special token id mismatches"):
            opd_validation.validate_teacher_student_tokenizer_compatibility(
                student_tokenizer=student, teacher_tokenizer=teacher, student_name="student", teacher_name="teacher"
            )


if __name__ == "__main__":
    unittest.main()
