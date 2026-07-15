from __future__ import annotations

from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from qwen_tts.cli.fastapi_service import (
    ADMIN_SHUTDOWN_HEADER,
    ADMIN_SHUTDOWN_VALUE,
    ServerShutdownController,
    ServiceSettings,
    create_app,
)
from qwen_tts.httpx_client import Qwen3TTSHttpxClient


class FakeModelManager:
    def loaded_models(self) -> list[str]:
        return []

    def get_model(self, kind: str):
        raise AssertionError(f"Shutdown tests must not load the {kind} model.")


def make_settings(root: Path) -> ServiceSettings:
    return ServiceSettings(
        storage_root=root,
        base_model="base",
        custom_model="custom",
        voice_design_model="design",
        device="mps",
        dtype="float32",
        attn_implementation="sdpa",
        narrator_speaker=None,
    )


def make_app(root: Path, callbacks: list[str], *, configured: bool = True):
    callback = (lambda: callbacks.append("shutdown")) if configured else None
    return create_app(
        settings=make_settings(root),
        model_manager=FakeModelManager(),
        shutdown_callback=callback,
    )


def test_shutdown_controller_drains_active_tts_requests_and_runs_once() -> None:
    callbacks: list[str] = []
    controller = ServerShutdownController(
        shutdown_callback=lambda: callbacks.append("shutdown"),
    )

    assert controller.begin_tts_request()
    assert controller.begin_tts_request()
    assert controller.request_shutdown("admin_request")
    assert not controller.begin_tts_request()
    assert not controller.execute_shutdown()

    controller.end_tts_request()
    assert callbacks == []
    controller.end_tts_request()
    assert callbacks == ["shutdown"]
    assert not controller.execute_shutdown()
    assert not controller.request_shutdown("admin_request")


def test_loopback_shutdown_is_accepted_and_callback_runs_once(tmp_path: Path) -> None:
    callbacks: list[str] = []
    app = make_app(tmp_path, callbacks)
    headers = {ADMIN_SHUTDOWN_HEADER: ADMIN_SHUTDOWN_VALUE}

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        first = client.post("/qwen3tts/admin/shutdown", headers=headers)
        second = client.post("/qwen3tts/admin/shutdown", headers=headers)

    assert first.status_code == 202
    assert first.json()["status"] == "accepted"
    assert second.status_code == 202
    assert second.json()["status"] == "already_pending"
    assert callbacks == ["shutdown"]


def test_ipv6_loopback_and_ipv4_mapped_loopback_are_accepted(tmp_path: Path) -> None:
    headers = {ADMIN_SHUTDOWN_HEADER: ADMIN_SHUTDOWN_VALUE}
    for host in ("::1", "::ffff:127.0.0.1"):
        callbacks: list[str] = []
        app = make_app(tmp_path / host.replace(":", "_"), callbacks)
        with TestClient(app, client=(host, 50000)) as client:
            response = client.post("/qwen3tts/admin/shutdown", headers=headers)
        assert response.status_code == 202
        assert callbacks == ["shutdown"]


def test_remote_client_cannot_spoof_loopback_with_forwarded_header(tmp_path: Path) -> None:
    callbacks: list[str] = []
    app = make_app(tmp_path, callbacks)
    headers = {
        ADMIN_SHUTDOWN_HEADER: ADMIN_SHUTDOWN_VALUE,
        "X-Forwarded-For": "127.0.0.1",
    }

    with TestClient(app, client=("192.168.1.50", 50000)) as client:
        response = client.post("/qwen3tts/admin/shutdown", headers=headers)

    assert response.status_code == 403
    assert callbacks == []


def test_shutdown_requires_confirmation_header(tmp_path: Path) -> None:
    callbacks: list[str] = []
    app = make_app(tmp_path, callbacks)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post("/qwen3tts/admin/shutdown")

    assert response.status_code == 403
    assert callbacks == []


def test_shutdown_requires_server_callback(tmp_path: Path) -> None:
    app = make_app(tmp_path, [], configured=False)
    headers = {ADMIN_SHUTDOWN_HEADER: ADMIN_SHUTDOWN_VALUE}

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post("/qwen3tts/admin/shutdown", headers=headers)

    assert response.status_code == 503


def test_pending_shutdown_rejects_new_tts_before_model_load(tmp_path: Path) -> None:
    callbacks: list[str] = []
    app = make_app(tmp_path, callbacks)
    assert app.state.shutdown_controller.request_shutdown("test")

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post(
            "/qwen3tts/tts/voice_clone",
            data={"text": "test", "ref_text": "reference"},
            files={"ref_audio": ("ref.wav", b"not-audio", "audio/wav")},
        )

    assert response.status_code == 503
    assert callbacks == []


def test_httpx_client_shutdown_sends_header_and_waits_for_exit() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/qwen3tts/admin/shutdown":
            return httpx.Response(
                202,
                json={"status": "accepted", "reason": "admin_request"},
            )
        raise httpx.ConnectError("server stopped", request=request)

    http_client = httpx.Client(
        base_url="http://127.0.0.1:8001",
        transport=httpx.MockTransport(handler),
    )
    with Qwen3TTSHttpxClient(client=http_client) as client:
        payload = client.shutdown(wait=True, wait_timeout=1)

    assert payload["server_stopped"] is True
    assert requests[0].url.path == "/qwen3tts/admin/shutdown"
    assert requests[0].headers[ADMIN_SHUTDOWN_HEADER] == ADMIN_SHUTDOWN_VALUE
    assert requests[0].headers["Connection"] == "close"
