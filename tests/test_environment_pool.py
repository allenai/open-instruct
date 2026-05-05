from open_instruct.environments.pool import _is_podman_host_failure


def test_is_podman_host_failure_detects_unresponsive_socket():
    error = RuntimeError(
        "Reset failed after 5 attempts: Error while fetching server API version: "
        "UnixHTTPConnectionPool(host='localhost', port=None): Read timed out. (read timeout=300)"
    )

    assert _is_podman_host_failure(error)


def test_is_podman_host_failure_detects_connection_errors():
    assert _is_podman_host_failure(ConnectionError("Connection refused while connecting to podman.sock"))


def test_is_podman_host_failure_ignores_task_failures():
    assert not _is_podman_host_failure(RuntimeError("Command timed out after 120s."))
