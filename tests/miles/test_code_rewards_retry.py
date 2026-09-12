"""The code-service session retries transient gateway failures on its POSTs."""

from open_instruct.miles import code_rewards


def test_retry_policy_covers_the_scoring_posts():
    retry = code_rewards.RETRY
    assert "POST" in retry.allowed_methods
    assert retry.total >= 5
    assert {500, 502, 503, 504} <= set(retry.status_forcelist)
    # Enough backoff to outlast a gateway hiccup, not so much that a dead service hangs a run forever.
    assert 60 <= sum(min(retry.backoff_factor * 2**i, 120) for i in range(retry.total)) <= 600


def test_session_mounts_the_retry_policy():
    session = code_rewards._get_session()
    assert session.get_adapter("https://example.invalid/").max_retries is code_rewards.RETRY
