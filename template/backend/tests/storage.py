"""A real RustFS for the storage tests: the image backend/compose.yaml uses, one per test session.
Each test gets its own bucket, so tests never see each other's objects."""

from __future__ import annotations

from collections.abc import Iterator

from tests.db import BACKEND

ACCESS_KEY = "test-storage"
SECRET_KEY = "test-storage-secret"  # noqa: S105 - a throwaway test container's key


def storage_image() -> str:
    import yaml

    compose = yaml.safe_load((BACKEND / "compose.yaml").read_text())
    image: str = compose["services"]["storage"]["image"]
    return image


def storage_endpoint() -> Iterator[str]:
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.wait_strategies import HttpWaitStrategy

    container = (
        DockerContainer(storage_image())
        .with_env("RUSTFS_ACCESS_KEY", ACCESS_KEY)
        .with_env("RUSTFS_SECRET_KEY", SECRET_KEY)
        .with_exposed_ports(9000)
        .waiting_for(HttpWaitStrategy(9000, "/health"))
    )
    with container:
        host, port = container.get_container_host_ip(), container.get_exposed_port(9000)
        yield f"http://{host}:{port}"
