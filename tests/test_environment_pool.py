from open_instruct.environments.pool import _actor_kwargs_for_slot


def test_actor_kwargs_for_slot_assigns_docker_host_round_robin():
    base_kwargs = {"backend": "docker", "image": "test-image"}

    kwargs = _actor_kwargs_for_slot(base_kwargs, 5, ["unix:///tmp/podman-0.sock", "unix:///tmp/podman-1.sock"])

    assert kwargs["docker_host"] == "unix:///tmp/podman-1.sock"
    assert "docker_host" not in base_kwargs


def test_actor_kwargs_for_slot_preserves_explicit_docker_host():
    base_kwargs = {"backend": "docker", "docker_host": "unix:///tmp/custom.sock"}

    kwargs = _actor_kwargs_for_slot(base_kwargs, 1, ["unix:///tmp/podman-0.sock", "unix:///tmp/podman-1.sock"])

    assert kwargs["docker_host"] == "unix:///tmp/custom.sock"


def test_actor_kwargs_for_slot_skips_non_docker_backends():
    kwargs = _actor_kwargs_for_slot({"backend": "apptainer"}, 0, ["unix:///tmp/podman-0.sock"])

    assert "docker_host" not in kwargs
