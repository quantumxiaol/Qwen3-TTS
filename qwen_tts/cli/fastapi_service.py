# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import os
import platform
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Optional

if platform.system() == "Darwin":
    # Configure MPS to fall back to CPU and avoid memory pressure on macOS.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import Qwen3TTSModel

DEFAULT_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_STORAGE_ROOT = Path("storage") / "qwen3_tts_service"
API_PREFIX = "/qwen3tts"
DEFAULT_NARRATOR_BY_LANGUAGE = {
    "chinese": "Uncle_Fu",
    "english": "Ryan",
    "japanese": "Ono_Anna",
    "korean": "Sohee",
}


def _safe_stem(value: str, fallback: str) -> str:
    stem = Path(value or "").stem
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem).strip("._")
    return cleaned or fallback


def _safe_filename(value: Optional[str], fallback_prefix: str = "audio") -> str:
    if value:
        name = Path(value).name
        stem = _safe_stem(name, fallback_prefix)
        return f"{stem}.wav"
    return f"{fallback_prefix}_{uuid.uuid4().hex}.wav"


def _resolve_torch_dtype(name: Optional[str], device_name: str) -> torch.dtype:
    if name:
        normalized = name.strip().lower()
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported dtype: {name}")
        return mapping[normalized]

    if device_name.startswith("cuda"):
        return torch.bfloat16
    if device_name.startswith("mps"):
        return torch.float16
    return torch.float32


def _choose_runtime(device_override: Optional[str], dtype_name: Optional[str], attn_override: Optional[str]) -> tuple[dict[str, Any], str]:
    if device_override:
        device_name = device_override
    elif torch.cuda.is_available():
        device_name = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    dtype = _resolve_torch_dtype(dtype_name, device_name)
    if attn_override:
        attn_implementation = attn_override
    elif device_name.startswith("cuda") and dtype in {torch.float16, torch.bfloat16}:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    return {
        "device_map": device_name,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
    }, device_name


def _build_gen_kwargs(
    max_new_tokens: Optional[int],
    do_sample: Optional[bool],
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: Optional[float],
    repetition_penalty: Optional[float],
    subtalker_top_k: Optional[int],
    subtalker_top_p: Optional[float],
    subtalker_temperature: Optional[float],
) -> dict[str, Any]:
    mapping = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "subtalker_top_k": subtalker_top_k,
        "subtalker_top_p": subtalker_top_p,
        "subtalker_temperature": subtalker_temperature,
    }
    return {key: value for key, value in mapping.items() if value is not None}


@dataclass
class ServiceSettings:
    storage_root: Path
    base_model: str
    custom_model: str
    voice_design_model: str
    device: Optional[str]
    dtype: Optional[str]
    attn_implementation: Optional[str]
    narrator_speaker: Optional[str]

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            storage_root=Path(os.getenv("QWEN3_TTS_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))).expanduser().resolve(),
            base_model=os.getenv("QWEN3_TTS_BASE_MODEL", DEFAULT_BASE_MODEL),
            custom_model=os.getenv("QWEN3_TTS_CUSTOM_MODEL", DEFAULT_CUSTOM_MODEL),
            voice_design_model=os.getenv("QWEN3_TTS_VOICE_DESIGN_MODEL", DEFAULT_VOICE_DESIGN_MODEL),
            device=os.getenv("QWEN3_TTS_DEVICE") or None,
            dtype=os.getenv("QWEN3_TTS_DTYPE") or None,
            attn_implementation=os.getenv("QWEN3_TTS_ATTN_IMPLEMENTATION") or None,
            narrator_speaker=os.getenv("QWEN3_TTS_NARRATOR_SPEAKER") or None,
        )


class StoredFile(BaseModel):
    filename: str
    path: str
    url: str


class VoiceCloneResponse(BaseModel):
    status: str
    request_id: str
    audio: StoredFile
    prompt_audio: StoredFile
    prompt_text: Optional[StoredFile]
    synthesis_text: StoredFile
    sample_rate: int
    language: str
    x_vector_only_mode: bool


class NarrationResponse(BaseModel):
    status: str
    request_id: str
    audio: StoredFile
    synthesis_text: StoredFile
    sample_rate: int
    language: str
    speaker: str
    instruct: Optional[str]


class VoiceCloneBatchResponse(BaseModel):
    status: str
    request_id: str
    audio_paths: list[StoredFile]
    prompt_audio: StoredFile
    prompt_text: Optional[StoredFile]
    text_file: StoredFile
    sample_rate: int
    language: str
    x_vector_only_mode: bool


class NarrationBatchResponse(BaseModel):
    status: str
    request_id: str
    audio_paths: list[StoredFile]
    text_file: StoredFile
    sample_rate: int
    language: str
    speaker: str
    instruct: Optional[str]


class HealthResponse(BaseModel):
    status: str
    storage_root: str
    loaded_models: list[str]


class NarratorCatalogResponse(BaseModel):
    status: str
    default_by_language: dict[str, str]
    supported_speakers: list[str]


class FileStore:
    def __init__(self, root: Path):
        self.root = root
        self.upload_root = root / "uploads"
        self.output_root = root / "outputs"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def request_upload_dir(self, request_id: str) -> Path:
        path = self.upload_root / request_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def request_output_dir(self, request_id: str) -> Path:
        path = self.output_root / request_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_upload(self, upload: UploadFile, request_id: str, fallback_prefix: str) -> Path:
        suffix = Path(upload.filename or "").suffix.lower() or ".bin"
        filename = f"{_safe_stem(upload.filename or fallback_prefix, fallback_prefix)}_{uuid.uuid4().hex[:8]}{suffix}"
        target = self.request_upload_dir(request_id) / filename
        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        await upload.close()
        return target.resolve()

    def save_text(self, text: str, request_id: str, filename: str) -> Path:
        target = self.request_upload_dir(request_id) / filename
        target.write_text(text, encoding="utf-8")
        return target.resolve()

    def build_output_path(self, request_id: str, output_name: Optional[str], prefix: str) -> Path:
        filename = _safe_filename(output_name, fallback_prefix=prefix)
        target = self.request_output_dir(request_id) / filename
        return target.resolve()

    def resolve_public_file(self, category: str, request_id: str, filename: str) -> Path:
        if category == "uploads":
            root = self.upload_root
        elif category == "outputs":
            root = self.output_root
        else:
            raise HTTPException(status_code=404, detail=f"Unknown file category: {category}")

        candidate = (root / request_id / filename).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Invalid file path.") from exc
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        return candidate


class ModelManager:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.runtime_kwargs, self.device_name = _choose_runtime(
            settings.device,
            settings.dtype,
            settings.attn_implementation,
        )
        self._models: dict[str, Qwen3TTSModel] = {}
        self._lock = threading.Lock()

    def loaded_models(self) -> list[str]:
        return sorted(self._models.keys())

    def _model_path(self, kind: str) -> str:
        if kind == "base":
            return self.settings.base_model
        if kind == "custom":
            return self.settings.custom_model
        if kind == "voice_design":
            return self.settings.voice_design_model
        raise ValueError(f"Unknown model kind: {kind}")

    def get_model(self, kind: str) -> Qwen3TTSModel:
        model = self._models.get(kind)
        if model is not None:
            return model

        with self._lock:
            model = self._models.get(kind)
            if model is not None:
                return model
            self._models[kind] = Qwen3TTSModel.from_pretrained(self._model_path(kind), **self.runtime_kwargs)
            return self._models[kind]


def _stored_file_from_path(request: Request, store: FileStore, path: Path) -> StoredFile:
    relative = path.resolve().relative_to(store.root.resolve())
    parts = relative.parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected stored path layout: {path}")
    url = str(
        request.url_for(
            "download_file",
            category=parts[0],
            request_id=parts[1],
            filename="/".join(parts[2:]),
        )
    )
    return StoredFile(filename=path.name, path=str(path.resolve()), url=url)


async def _resolve_required_text(
    store: FileStore,
    request_id: str,
    text: Optional[str],
    text_file: Optional[UploadFile],
    default_filename: str,
) -> tuple[str, Path]:
    if text_file is not None:
        saved_path = await store.save_upload(text_file, request_id, fallback_prefix=Path(default_filename).stem)
        normalized = saved_path.read_text(encoding="utf-8").strip()
        if not normalized:
            raise HTTPException(status_code=422, detail=f"{default_filename} is empty.")
        return normalized, saved_path
    if text is not None and text.strip():
        normalized = text.strip()
        return normalized, store.save_text(normalized, request_id, default_filename)
    raise HTTPException(status_code=422, detail="Either text or text_file must be provided.")


async def _resolve_optional_text(
    store: FileStore,
    request_id: str,
    text: Optional[str],
    text_file: Optional[UploadFile],
    default_filename: str,
) -> tuple[Optional[str], Optional[Path]]:
    if text_file is not None:
        saved_path = await store.save_upload(text_file, request_id, fallback_prefix=Path(default_filename).stem)
        normalized = saved_path.read_text(encoding="utf-8").strip()
        if not normalized:
            raise HTTPException(status_code=422, detail=f"{default_filename} is empty.")
        return normalized, saved_path
    if text is not None and text.strip():
        normalized = text.strip()
        return normalized, store.save_text(normalized, request_id, default_filename)
    return None, None


async def _resolve_required_text_file(
    store: FileStore,
    request_id: str,
    text_file: Optional[UploadFile],
    default_filename: str,
) -> tuple[Path, list[tuple[int, str]]]:
    if text_file is None:
        raise HTTPException(status_code=422, detail="text_file is required.")

    saved_path = await store.save_upload(text_file, request_id, fallback_prefix=Path(default_filename).stem)
    lines: list[tuple[int, str]] = []
    with saved_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if text:
                lines.append((line_no, text))
    if not lines:
        raise HTTPException(status_code=422, detail=f"{default_filename} does not contain any non-empty lines.")
    return saved_path, lines


def _choose_default_narrator(language: str, narrator_speaker: Optional[str], supported: list[str]) -> str:
    if narrator_speaker:
        return narrator_speaker

    normalized_supported = {speaker.lower(): speaker for speaker in supported}
    candidate = DEFAULT_NARRATOR_BY_LANGUAGE.get((language or "auto").strip().lower())
    if candidate and candidate.lower() in normalized_supported:
        return normalized_supported[candidate.lower()]
    if "uncle_fu" in normalized_supported:
        return normalized_supported["uncle_fu"]
    if supported:
        return supported[0]
    raise HTTPException(status_code=503, detail="No supported narration speakers are available.")


def _resolve_batch_output_base(text_file_path: Path, output_prefix: Optional[str], fallback_prefix: str) -> str:
    if output_prefix:
        return _safe_stem(output_prefix, fallback_prefix)
    return _safe_stem(text_file_path.name, fallback_prefix)


def create_app(
    settings: Optional[ServiceSettings] = None,
    model_manager: Optional[ModelManager] = None,
) -> FastAPI:
    app_settings = settings or ServiceSettings.from_env()
    manager = model_manager or ModelManager(app_settings)
    store = FileStore(app_settings.storage_root)

    app = FastAPI(title="Qwen3-TTS FastAPI Service")
    app.state.settings = app_settings
    app.state.model_manager = manager
    app.state.file_store = store

    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            storage_root=str(store.root),
            loaded_models=manager.loaded_models(),
        )

    @app.get(f"{API_PREFIX}/tts/narrators", response_model=NarratorCatalogResponse)
    def list_narrators() -> NarratorCatalogResponse:
        model = manager.get_model("custom")
        speakers = model.get_supported_speakers() or []
        return NarratorCatalogResponse(
            status="success",
            default_by_language=DEFAULT_NARRATOR_BY_LANGUAGE,
            supported_speakers=speakers,
        )

    @app.get(f"{API_PREFIX}/files/{{category}}/{{request_id}}/{{filename:path}}", name="download_file")
    def download_file(category: str, request_id: str, filename: str) -> FileResponse:
        target = store.resolve_public_file(category, request_id, filename)
        return FileResponse(target)

    @app.post(f"{API_PREFIX}/tts/voice_clone", response_model=VoiceCloneResponse)
    async def tts_voice_clone(
        request: Request,
        ref_audio: Annotated[UploadFile, File(...)],
        text: Annotated[Optional[str], Form()] = None,
        text_file: Annotated[Optional[UploadFile], File()] = None,
        ref_text: Annotated[Optional[str], Form()] = None,
        ref_text_file: Annotated[Optional[UploadFile], File()] = None,
        language: Annotated[str, Form()] = "Auto",
        output_name: Annotated[Optional[str], Form()] = None,
        x_vector_only_mode: Annotated[bool, Form()] = False,
        max_new_tokens: Annotated[Optional[int], Form()] = None,
        do_sample: Annotated[Optional[bool], Form()] = None,
        top_k: Annotated[Optional[int], Form()] = None,
        top_p: Annotated[Optional[float], Form()] = None,
        temperature: Annotated[Optional[float], Form()] = None,
        repetition_penalty: Annotated[Optional[float], Form()] = None,
        subtalker_top_k: Annotated[Optional[int], Form()] = None,
        subtalker_top_p: Annotated[Optional[float], Form()] = None,
        subtalker_temperature: Annotated[Optional[float], Form()] = None,
    ) -> VoiceCloneResponse:
        request_id = uuid.uuid4().hex
        synthesis_text, synthesis_text_path = await _resolve_required_text(
            store,
            request_id,
            text,
            text_file,
            "synthesis_text.txt",
        )
        reference_text, reference_text_path = await _resolve_optional_text(
            store,
            request_id,
            ref_text,
            ref_text_file,
            "reference_text.txt",
        )
        if not x_vector_only_mode and not reference_text:
            raise HTTPException(status_code=422, detail="ref_text or ref_text_file is required when x_vector_only_mode=false.")

        prompt_audio_path = await store.save_upload(ref_audio, request_id, fallback_prefix="prompt_audio")
        output_path = store.build_output_path(request_id, output_name, prefix="voice_clone")

        model = manager.get_model("base")
        gen_kwargs = _build_gen_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
        )
        try:
            wavs, sr = model.generate_voice_clone(
                text=synthesis_text,
                language=language,
                ref_audio=str(prompt_audio_path),
                ref_text=reference_text,
                x_vector_only_mode=x_vector_only_mode,
                **gen_kwargs,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        sf.write(output_path, wavs[0], sr)
        return VoiceCloneResponse(
            status="success",
            request_id=request_id,
            audio=_stored_file_from_path(request, store, output_path),
            prompt_audio=_stored_file_from_path(request, store, prompt_audio_path),
            prompt_text=_stored_file_from_path(request, store, reference_text_path) if reference_text_path else None,
            synthesis_text=_stored_file_from_path(request, store, synthesis_text_path),
            sample_rate=int(sr),
            language=language,
            x_vector_only_mode=x_vector_only_mode,
        )

    @app.post(f"{API_PREFIX}/tts/voice_clone_batch_file", response_model=VoiceCloneBatchResponse)
    async def tts_voice_clone_batch_file(
        request: Request,
        ref_audio: Annotated[UploadFile, File(...)],
        text_file: Annotated[UploadFile, File(...)],
        ref_text: Annotated[Optional[str], Form()] = None,
        ref_text_file: Annotated[Optional[UploadFile], File()] = None,
        language: Annotated[str, Form()] = "Auto",
        output_prefix: Annotated[Optional[str], Form()] = None,
        x_vector_only_mode: Annotated[bool, Form()] = False,
        max_new_tokens: Annotated[Optional[int], Form()] = None,
        do_sample: Annotated[Optional[bool], Form()] = None,
        top_k: Annotated[Optional[int], Form()] = None,
        top_p: Annotated[Optional[float], Form()] = None,
        temperature: Annotated[Optional[float], Form()] = None,
        repetition_penalty: Annotated[Optional[float], Form()] = None,
        subtalker_top_k: Annotated[Optional[int], Form()] = None,
        subtalker_top_p: Annotated[Optional[float], Form()] = None,
        subtalker_temperature: Annotated[Optional[float], Form()] = None,
    ) -> VoiceCloneBatchResponse:
        request_id = uuid.uuid4().hex
        synthesis_text_path, lines = await _resolve_required_text_file(
            store,
            request_id,
            text_file,
            "synthesis_text.txt",
        )
        reference_text, reference_text_path = await _resolve_optional_text(
            store,
            request_id,
            ref_text,
            ref_text_file,
            "reference_text.txt",
        )
        if not x_vector_only_mode and not reference_text:
            raise HTTPException(status_code=422, detail="ref_text or ref_text_file is required when x_vector_only_mode=false.")

        prompt_audio_path = await store.save_upload(ref_audio, request_id, fallback_prefix="prompt_audio")
        output_base = _resolve_batch_output_base(synthesis_text_path, output_prefix, "voice_clone")

        model = manager.get_model("base")
        gen_kwargs = _build_gen_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
        )
        audio_paths: list[StoredFile] = []
        sample_rate: Optional[int] = None
        for line_no, line_text in lines:
            output_name = f"{output_base}_{line_no}.wav"
            output_path = store.build_output_path(request_id, output_name, prefix="voice_clone")
            try:
                wavs, sr = model.generate_voice_clone(
                    text=line_text,
                    language=language,
                    ref_audio=str(prompt_audio_path),
                    ref_text=reference_text,
                    x_vector_only_mode=x_vector_only_mode,
                    **gen_kwargs,
                )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            sf.write(output_path, wavs[0], sr)
            sample_rate = int(sr)
            audio_paths.append(_stored_file_from_path(request, store, output_path))

        return VoiceCloneBatchResponse(
            status="success",
            request_id=request_id,
            audio_paths=audio_paths,
            prompt_audio=_stored_file_from_path(request, store, prompt_audio_path),
            prompt_text=_stored_file_from_path(request, store, reference_text_path) if reference_text_path else None,
            text_file=_stored_file_from_path(request, store, synthesis_text_path),
            sample_rate=sample_rate or 0,
            language=language,
            x_vector_only_mode=x_vector_only_mode,
        )

    @app.post(f"{API_PREFIX}/tts/narration", response_model=NarrationResponse)
    async def tts_narration(
        request: Request,
        text: Annotated[Optional[str], Form()] = None,
        text_file: Annotated[Optional[UploadFile], File()] = None,
        language: Annotated[str, Form()] = "Auto",
        speaker: Annotated[Optional[str], Form()] = None,
        instruct: Annotated[Optional[str], Form()] = None,
        output_name: Annotated[Optional[str], Form()] = None,
        max_new_tokens: Annotated[Optional[int], Form()] = None,
        do_sample: Annotated[Optional[bool], Form()] = None,
        top_k: Annotated[Optional[int], Form()] = None,
        top_p: Annotated[Optional[float], Form()] = None,
        temperature: Annotated[Optional[float], Form()] = None,
        repetition_penalty: Annotated[Optional[float], Form()] = None,
        subtalker_top_k: Annotated[Optional[int], Form()] = None,
        subtalker_top_p: Annotated[Optional[float], Form()] = None,
        subtalker_temperature: Annotated[Optional[float], Form()] = None,
    ) -> NarrationResponse:
        request_id = uuid.uuid4().hex
        synthesis_text, synthesis_text_path = await _resolve_required_text(
            store,
            request_id,
            text,
            text_file,
            "narration_text.txt",
        )
        output_path = store.build_output_path(request_id, output_name, prefix="narration")

        model = manager.get_model("custom")
        supported_speakers = model.get_supported_speakers() or []
        resolved_speaker = speaker or _choose_default_narrator(
            language=language,
            narrator_speaker=app_settings.narrator_speaker,
            supported=supported_speakers,
        )
        gen_kwargs = _build_gen_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
        )
        try:
            wavs, sr = model.generate_custom_voice(
                text=synthesis_text,
                language=language,
                speaker=resolved_speaker,
                instruct=instruct or "",
                **gen_kwargs,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        sf.write(output_path, wavs[0], sr)
        return NarrationResponse(
            status="success",
            request_id=request_id,
            audio=_stored_file_from_path(request, store, output_path),
            synthesis_text=_stored_file_from_path(request, store, synthesis_text_path),
            sample_rate=int(sr),
            language=language,
            speaker=resolved_speaker,
            instruct=instruct,
        )

    @app.post(f"{API_PREFIX}/tts/narration_batch_file", response_model=NarrationBatchResponse)
    async def tts_narration_batch_file(
        request: Request,
        text_file: Annotated[UploadFile, File(...)],
        language: Annotated[str, Form()] = "Auto",
        speaker: Annotated[Optional[str], Form()] = None,
        instruct: Annotated[Optional[str], Form()] = None,
        output_prefix: Annotated[Optional[str], Form()] = None,
        max_new_tokens: Annotated[Optional[int], Form()] = None,
        do_sample: Annotated[Optional[bool], Form()] = None,
        top_k: Annotated[Optional[int], Form()] = None,
        top_p: Annotated[Optional[float], Form()] = None,
        temperature: Annotated[Optional[float], Form()] = None,
        repetition_penalty: Annotated[Optional[float], Form()] = None,
        subtalker_top_k: Annotated[Optional[int], Form()] = None,
        subtalker_top_p: Annotated[Optional[float], Form()] = None,
        subtalker_temperature: Annotated[Optional[float], Form()] = None,
    ) -> NarrationBatchResponse:
        request_id = uuid.uuid4().hex
        synthesis_text_path, lines = await _resolve_required_text_file(
            store,
            request_id,
            text_file,
            "narration_text.txt",
        )
        output_base = _resolve_batch_output_base(synthesis_text_path, output_prefix, "narration")

        model = manager.get_model("custom")
        supported_speakers = model.get_supported_speakers() or []
        resolved_speaker = speaker or _choose_default_narrator(
            language=language,
            narrator_speaker=app_settings.narrator_speaker,
            supported=supported_speakers,
        )
        gen_kwargs = _build_gen_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
        )
        audio_paths: list[StoredFile] = []
        sample_rate: Optional[int] = None
        for line_no, line_text in lines:
            output_name = f"{output_base}_{line_no}.wav"
            output_path = store.build_output_path(request_id, output_name, prefix="narration")
            try:
                wavs, sr = model.generate_custom_voice(
                    text=line_text,
                    language=language,
                    speaker=resolved_speaker,
                    instruct=instruct or "",
                    **gen_kwargs,
                )
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            sf.write(output_path, wavs[0], sr)
            sample_rate = int(sr)
            audio_paths.append(_stored_file_from_path(request, store, output_path))

        return NarrationBatchResponse(
            status="success",
            request_id=request_id,
            audio_paths=audio_paths,
            text_file=_stored_file_from_path(request, store, synthesis_text_path),
            sample_rate=sample_rate or 0,
            language=language,
            speaker=resolved_speaker,
            instruct=instruct,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-server",
        description="Launch a FastAPI server for Qwen3-TTS voice clone and narration.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001).")
    parser.add_argument("--storage-root", default=None, help="Directory for uploaded files and generated outputs.")
    parser.add_argument("--base-model", default=None, help="Base model path or Hugging Face id.")
    parser.add_argument("--custom-model", default=None, help="CustomVoice model path or Hugging Face id.")
    parser.add_argument("--voice-design-model", default=None, help="VoiceDesign model path or Hugging Face id.")
    parser.add_argument("--device", default=None, help="device_map value, such as cpu/cuda:0/mps.")
    parser.add_argument("--dtype", default=None, help="Torch dtype: bfloat16/float16/float32.")
    parser.add_argument("--attn-implementation", default=None, help="Attention backend override.")
    parser.add_argument("--narrator-speaker", default=None, help="Default speaker for the narration endpoint.")
    return parser


app = create_app()
