from __future__ import annotations

import tempfile
from pathlib import Path

import httpx

from qwen_tts.httpx_client import Qwen3TTSHttpxClient


def test_httpx_client_voice_clone_request_and_download() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        captured["path"] = request.url.path
        captured["body"] = body

        if request.url.path == "/qwen3tts/tts/voice_clone":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "audio": {
                        "url": "http://testserver/qwen3tts/files/outputs/req/output.wav",
                    },
                },
            )
        if request.url.path == "/qwen3tts/files/outputs/req/output.wav":
            return httpx.Response(200, content=b"wav-bytes")
        raise AssertionError(f"Unexpected path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"audio")
        output_path = tmp_path / "downloaded.wav"

        with Qwen3TTSHttpxClient(client=httpx.Client(base_url="http://testserver", transport=transport)) as client:
            payload = client.voice_clone(
                ref_audio_path=ref_audio,
                text="你好",
                ref_text="参考文本",
                language="Chinese",
                output_name="demo",
                download_to=output_path,
            )

        assert payload["status"] == "success"
        assert captured["path"] == "/qwen3tts/tts/voice_clone"
        body = captured["body"]
        assert isinstance(body, bytes)
        assert b'name="text"' in body
        assert b'name="ref_text"' in body
        assert b'name="ref_audio"' in body
        assert output_path.read_bytes() == b"wav-bytes"


def test_httpx_client_narration_and_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/qwen3tts/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/qwen3tts/tts/narrators":
            return httpx.Response(200, json={"status": "success", "supported_speakers": ["Uncle_Fu"]})
        if request.url.path == "/qwen3tts/tts/narration":
            return httpx.Response(200, json={"status": "success", "speaker": "Uncle_Fu"})
        raise AssertionError(f"Unexpected path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    with Qwen3TTSHttpxClient(client=httpx.Client(base_url="http://testserver", transport=transport)) as client:
        assert client.health()["status"] == "ok"
        assert client.list_narrators()["status"] == "success"
        assert client.narration(text="旁白", language="Chinese")["speaker"] == "Uncle_Fu"


def test_httpx_client_batch_endpoints_and_download_dir() -> None:
    captured_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        if request.url.path == "/qwen3tts/tts/narration_batch_file":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "audio_paths": [
                        {"filename": "a.wav", "url": "http://testserver/qwen3tts/files/outputs/req/a.wav"},
                        {"filename": "b.wav", "url": "http://testserver/qwen3tts/files/outputs/req/b.wav"},
                    ],
                },
            )
        if request.url.path == "/qwen3tts/tts/voice_clone_batch_file":
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "audio_paths": [
                        {"filename": "c.wav", "url": "http://testserver/qwen3tts/files/outputs/req/c.wav"},
                    ],
                },
            )
        if request.url.path.endswith("/a.wav"):
            return httpx.Response(200, content=b"a")
        if request.url.path.endswith("/b.wav"):
            return httpx.Response(200, content=b"b")
        if request.url.path.endswith("/c.wav"):
            return httpx.Response(200, content=b"c")
        raise AssertionError(f"Unexpected path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"audio")
        lines = tmp_path / "lines.txt"
        lines.write_text("第一句。\n第二句。\n", encoding="utf-8")
        batch_download_dir = tmp_path / "batch_downloads"
        clone_download_dir = tmp_path / "clone_downloads"

        with Qwen3TTSHttpxClient(client=httpx.Client(base_url="http://testserver", transport=transport)) as client:
            narration_payload = client.narration_batch_file(
                text_file=lines,
                language="Chinese",
                download_dir=batch_download_dir,
            )
            clone_payload = client.voice_clone_batch_file(
                ref_audio_path=ref_audio,
                text_file=lines,
                ref_text="参考文本",
                language="Chinese",
                download_dir=clone_download_dir,
            )

        assert narration_payload["status"] == "success"
        assert clone_payload["status"] == "success"
        assert (batch_download_dir / "a.wav").read_bytes() == b"a"
        assert (batch_download_dir / "b.wav").read_bytes() == b"b"
        assert (clone_download_dir / "c.wav").read_bytes() == b"c"
        assert "/qwen3tts/tts/narration_batch_file" in captured_paths
        assert "/qwen3tts/tts/voice_clone_batch_file" in captured_paths
