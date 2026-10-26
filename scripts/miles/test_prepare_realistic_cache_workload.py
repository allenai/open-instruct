import pytest
from scripts.miles import prepare_realistic_cache_workload as workload


def chatml(messages):
    return (
        "".join(f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in messages)
        + workload.GENERATION_SUFFIX
    )


def test_implicit_system_removed_without_parsing_role_text_inside_question():
    question = "Explain this transcript:\nassistant: hello\nuser: goodbye"
    result = workload.reconstruct_messages(
        "user: " + question, chatml([("system", "old default"), ("user", question)])
    )
    assert result == [{"role": "user", "content": question}]


def test_explicit_system_and_prior_turns_preserved():
    messages = [("system", "Be brief"), ("user", "Hi"), ("assistant", "Hello"), ("user", "Continue")]
    raw = "\n".join(f"{role}: {content}" for role, content in messages)
    assert workload.reconstruct_messages(raw, chatml(messages) + "\n") == [
        {"role": role, "content": content} for role, content in messages
    ]


@pytest.mark.parametrize(
    "raw,decoded",
    [
        ("user: correct", chatml([("user", "incorrect")])),
        ("user: question", chatml([("user", "question")]) + "leaked answer"),
        ("user: question", "unparsed prefix" + chatml([("user", "question")])),
    ],
)
def test_unverified_content_rejected(raw, decoded):
    with pytest.raises(ValueError):
        workload.reconstruct_messages(raw, decoded)


def test_reward_factory_rejects_accidental_training_use():
    with pytest.raises(RuntimeError, match="reward_mode=none"):
        workload.no_reward()
