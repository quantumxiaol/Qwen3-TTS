from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from qwen_tts.cli.fastapi_service import ServiceSettings, create_app


class FakeBaseModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_voice_clone(self, **kwargs):
        self.calls.append(kwargs)
        return [np.zeros(2400, dtype=np.float32)], 24000


class FakeCustomModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_supported_speakers(self):
        return ["Uncle_Fu", "Ryan", "Vivian"]

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        return [np.ones(2400, dtype=np.float32) * 0.1], 24000


class FakeModelManager:
    def __init__(self) -> None:
        self.base = FakeBaseModel()
        self.custom = FakeCustomModel()

    def loaded_models(self) -> list[str]:
        return []

    def get_model(self, kind: str):
        if kind == "base":
            return self.base
        if kind == "custom":
            return self.custom
        raise AssertionError(f"Unexpected model kind: {kind}")


def test_voice_clone_endpoint_saves_inputs_and_outputs(tmp_path: Path) -> None:
    settings = ServiceSettings(
        storage_root=tmp_path / "service_storage",
        base_model="base",
        custom_model="custom",
        voice_design_model="design",
        device="cpu",
        dtype="float32",
        attn_implementation="sdpa",
        narrator_speaker=None,
    )
    manager = FakeModelManager()
    app = create_app(settings=settings, model_manager=manager)
    client = TestClient(app)

    response = client.post(
        "/qwen3tts/tts/voice_clone",
        data={
            "text": "你好，这是克隆测试。",
            "ref_text": "这是参考音频对应的文本。",
            "language": "Chinese",
            "output_name": "demo_clone",
        },
        files={
            "ref_audio": ("speaker.wav", b"fake-audio", "audio/wav"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["audio"]["path"].endswith("demo_clone.wav")
    assert Path(payload["audio"]["path"]).exists()
    assert Path(payload["prompt_audio"]["path"]).exists()
    assert Path(payload["prompt_text"]["path"]).exists()
    assert Path(payload["synthesis_text"]["path"]).exists()

    call = manager.base.calls[0]
    assert Path(call["ref_audio"]).exists()
    assert call["ref_text"] == "这是参考音频对应的文本。"
    assert call["language"] == "Chinese"
    assert call["text"] == "你好，这是克隆测试。"

    download_response = client.get(payload["audio"]["url"])
    assert download_response.status_code == 200
    assert len(download_response.content) > 0


def test_narration_endpoint_uses_default_speaker_and_text_file(tmp_path: Path) -> None:
    settings = ServiceSettings(
        storage_root=tmp_path / "service_storage",
        base_model="base",
        custom_model="custom",
        voice_design_model="design",
        device="cpu",
        dtype="float32",
        attn_implementation="sdpa",
        narrator_speaker=None,
    )
    manager = FakeModelManager()
    app = create_app(settings=settings, model_manager=manager)
    client = TestClient(app)

    response = client.post(
        "/qwen3tts/tts/narration",
        data={
            "language": "Chinese",
            "instruct": "保持平稳的纪录片旁白语气。",
        },
        files={
            "text_file": ("narration.txt", "这是需要生成的旁白文本。".encode("utf-8"), "text/plain"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["speaker"] == "Uncle_Fu"
    assert Path(payload["audio"]["path"]).exists()
    assert Path(payload["synthesis_text"]["path"]).exists()

    call = manager.custom.calls[0]
    assert call["speaker"] == "Uncle_Fu"
    assert call["language"] == "Chinese"
    assert call["instruct"] == "保持平稳的纪录片旁白语气。"
    assert call["text"] == "这是需要生成的旁白文本。"


def test_voice_clone_batch_file_endpoint_generates_one_wav_per_nonempty_line(tmp_path: Path) -> None:
    settings = ServiceSettings(
        storage_root=tmp_path / "service_storage",
        base_model="base",
        custom_model="custom",
        voice_design_model="design",
        device="cpu",
        dtype="float32",
        attn_implementation="sdpa",
        narrator_speaker=None,
    )
    manager = FakeModelManager()
    app = create_app(settings=settings, model_manager=manager)
    client = TestClient(app)

    response = client.post(
        "/qwen3tts/tts/voice_clone_batch_file",
        data={
            "ref_text": "这是参考音频对应的文本。",
            "language": "Chinese",
            "output_prefix": "batch_clone",
        },
        files={
            "ref_audio": ("speaker.wav", b"fake-audio", "audio/wav"),
            "text_file": ("lines.txt", "第一句。\n\n第二句。\n".encode("utf-8"), "text/plain"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert len(payload["audio_paths"]) == 2
    assert payload["audio_paths"][0]["filename"] == "batch_clone_1.wav"
    assert payload["audio_paths"][1]["filename"] == "batch_clone_3.wav"
    assert Path(payload["text_file"]["path"]).exists()
    assert len(manager.base.calls) == 2
    assert manager.base.calls[0]["text"] == "第一句。"
    assert manager.base.calls[1]["text"] == "第二句。"


def test_narration_batch_file_endpoint_generates_one_wav_per_nonempty_line(tmp_path: Path) -> None:
    settings = ServiceSettings(
        storage_root=tmp_path / "service_storage",
        base_model="base",
        custom_model="custom",
        voice_design_model="design",
        device="cpu",
        dtype="float32",
        attn_implementation="sdpa",
        narrator_speaker=None,
    )
    manager = FakeModelManager()
    app = create_app(settings=settings, model_manager=manager)
    client = TestClient(app)

    response = client.post(
        "/qwen3tts/tts/narration_batch_file",
        data={
            "language": "Chinese",
            "output_prefix": "narration_batch",
            "instruct": "保持平稳的纪录片旁白语气。",
        },
        files={
            "text_file": ("lines.txt", "第一句。\n第二句。\n".encode("utf-8"), "text/plain"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert len(payload["audio_paths"]) == 2
    assert payload["audio_paths"][0]["filename"] == "narration_batch_1.wav"
    assert payload["audio_paths"][1]["filename"] == "narration_batch_2.wav"
    assert payload["speaker"] == "Uncle_Fu"
    assert len(manager.custom.calls) == 2
    assert manager.custom.calls[0]["text"] == "第一句。"
    assert manager.custom.calls[1]["text"] == "第二句。"
