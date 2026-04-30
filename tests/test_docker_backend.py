"""Unit tests for DockerBackend retry behavior."""

from unittest.mock import MagicMock, patch

import docker as docker_sdk

from open_instruct.environments.backends import DockerBackend


class _FakeDockerContainer:
    short_id = "abc123"

    def __init__(self, *, exec_errors=None):
        self.attrs = {"State": {"Status": "running", "Running": True, "ExitCode": 0, "OOMKilled": False, "Error": ""}}
        self.exec_calls = 0
        self.reload_calls = 0
        self.exec_errors = list(exec_errors or [])

    def exec_run(self, *_args, **_kwargs):
        self.exec_calls += 1
        if self.exec_errors:
            raise self.exec_errors.pop(0)
        return 0, (b"ok\n", b"")

    def reload(self):
        self.reload_calls += 1


def _api_error(message: str) -> docker_sdk.errors.APIError:
    return docker_sdk.errors.APIError(message)


def test_create_client_uses_explicit_docker_host():
    backend = DockerBackend(image="test-image", docker_host="unix:///tmp/podman-1.sock")
    expected_client = MagicMock()

    with (
        patch(
            "open_instruct.environments.backends.docker_sdk.DockerClient", return_value=expected_client
        ) as docker_client,
        patch("open_instruct.environments.backends.docker_sdk.from_env") as from_env,
    ):
        client = backend._create_client()

    assert client is expected_client
    docker_client.assert_called_once_with(base_url="unix:///tmp/podman-1.sock", timeout=300)
    from_env.assert_not_called()


def test_create_client_uses_environment_without_explicit_docker_host():
    backend = DockerBackend(image="test-image")
    expected_client = MagicMock()

    with (
        patch("open_instruct.environments.backends.docker_sdk.DockerClient") as docker_client,
        patch("open_instruct.environments.backends.docker_sdk.from_env", return_value=expected_client) as from_env,
    ):
        client = backend._create_client()

    assert client is expected_client
    from_env.assert_called_once_with(timeout=300)
    docker_client.assert_not_called()


def test_run_command_retries_database_locked_exec_error_on_same_container():
    container = _FakeDockerContainer(exec_errors=[_api_error("database is locked")])
    backend = DockerBackend(image="test-image")
    backend._container = container
    backend._client = MagicMock()
    backend._client.containers.get.return_value = container

    with (
        patch("open_instruct.environments.backends.random.uniform", return_value=0.25),
        patch("open_instruct.environments.backends.time.sleep") as sleep,
        patch.object(backend, "start") as start,
    ):
        result = backend.run_command("echo ok")

    assert result.stdout == "ok\n"
    assert result.exit_code == 0
    assert container.exec_calls == 2
    assert container.reload_calls == 2
    sleep.assert_called_once_with(0.75)
    start.assert_not_called()


def test_run_command_allows_five_database_locked_exec_retries():
    container = _FakeDockerContainer(exec_errors=[_api_error("database is locked") for _ in range(5)])
    backend = DockerBackend(image="test-image")
    backend._container = container
    backend._client = MagicMock()
    backend._client.containers.get.return_value = container

    with (
        patch("open_instruct.environments.backends.random.uniform", return_value=0.25),
        patch("open_instruct.environments.backends.time.sleep") as sleep,
    ):
        result = backend.run_command("echo ok")

    assert result.stdout == "ok\n"
    assert result.exit_code == 0
    assert container.exec_calls == 6
    assert container.reload_calls == 6
    assert [call.args[0] for call in sleep.call_args_list] == [0.75, 1.25, 2.25, 4.25, 8.25]


def test_run_command_does_not_retry_unknown_exec_api_error():
    container = _FakeDockerContainer(exec_errors=[_api_error("unknown docker failure")])
    backend = DockerBackend(image="test-image")
    backend._container = container
    backend._client = MagicMock()
    backend._client.containers.get.return_value = container

    try:
        backend.run_command("echo ok")
    except docker_sdk.errors.APIError:
        pass
    else:
        raise AssertionError("Expected APIError to be raised")

    assert container.exec_calls == 1
