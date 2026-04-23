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

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional

import httpx

API_PREFIX = "/qwen3tts"
DEFAULT_SERVER_URL = "http://127.0.0.1:8001"


def _form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class Qwen3TTSHttpxClient:
    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        timeout: float = 300.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        normalized_server_url = server_url.rstrip("/")
        if normalized_server_url.endswith(API_PREFIX):
            normalized_server_url = normalized_server_url[: -len(API_PREFIX)]
        self.server_url = normalized_server_url
        self._client = client or httpx.Client(base_url=self.server_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Qwen3TTSHttpxClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._json(self._client.get(f"{API_PREFIX}/health"))

    def list_narrators(self) -> dict[str, Any]:
        return self._json(self._client.get(f"{API_PREFIX}/tts/narrators"))

    def download_url(self, url: str, output_path: str | Path) -> Path:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._client.stream("GET", url) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return target

    def voice_clone(
        self,
        *,
        ref_audio_path: str | Path,
        text: Optional[str] = None,
        text_file: Optional[str | Path] = None,
        ref_text: Optional[str] = None,
        ref_text_file: Optional[str | Path] = None,
        language: str = "Auto",
        output_name: Optional[str] = None,
        x_vector_only_mode: bool = False,
        download_to: Optional[str | Path] = None,
        **gen_kwargs,
    ) -> dict[str, Any]:
        with ExitStack() as stack:
            data: dict[str, str] = {"language": language, "x_vector_only_mode": _form_value(x_vector_only_mode)}
            files: dict[str, tuple[str, Any, str]] = {}

            if text is not None:
                data["text"] = text
            if ref_text is not None:
                data["ref_text"] = ref_text
            if output_name is not None:
                data["output_name"] = output_name
            for key, value in gen_kwargs.items():
                if value is not None:
                    data[key] = _form_value(value)

            ref_audio = Path(ref_audio_path).expanduser().resolve()
            files["ref_audio"] = (ref_audio.name, stack.enter_context(ref_audio.open("rb")), "audio/*")

            if text_file is not None:
                path = Path(text_file).expanduser().resolve()
                files["text_file"] = (path.name, stack.enter_context(path.open("rb")), "text/plain")

            if ref_text_file is not None:
                path = Path(ref_text_file).expanduser().resolve()
                files["ref_text_file"] = (path.name, stack.enter_context(path.open("rb")), "text/plain")

            response = self._client.post(f"{API_PREFIX}/tts/voice_clone", data=data, files=files)

        payload = self._json(response)
        if download_to is not None:
            self.download_url(payload["audio"]["url"], download_to)
        return payload

    def narration(
        self,
        *,
        text: Optional[str] = None,
        text_file: Optional[str | Path] = None,
        language: str = "Auto",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        output_name: Optional[str] = None,
        download_to: Optional[str | Path] = None,
        **gen_kwargs,
    ) -> dict[str, Any]:
        with ExitStack() as stack:
            data: dict[str, str] = {"language": language}
            files: dict[str, tuple[str, Any, str]] = {}

            if text is not None:
                data["text"] = text
            if speaker is not None:
                data["speaker"] = speaker
            if instruct is not None:
                data["instruct"] = instruct
            if output_name is not None:
                data["output_name"] = output_name
            for key, value in gen_kwargs.items():
                if value is not None:
                    data[key] = _form_value(value)

            if text_file is not None:
                path = Path(text_file).expanduser().resolve()
                files["text_file"] = (path.name, stack.enter_context(path.open("rb")), "text/plain")

            response = self._client.post(f"{API_PREFIX}/tts/narration", data=data, files=files)

        payload = self._json(response)
        if download_to is not None:
            self.download_url(payload["audio"]["url"], download_to)
        return payload

    def voice_clone_batch_file(
        self,
        *,
        ref_audio_path: str | Path,
        text_file: str | Path,
        ref_text: Optional[str] = None,
        ref_text_file: Optional[str | Path] = None,
        language: str = "Auto",
        output_prefix: Optional[str] = None,
        x_vector_only_mode: bool = False,
        download_dir: Optional[str | Path] = None,
        **gen_kwargs,
    ) -> dict[str, Any]:
        with ExitStack() as stack:
            data: dict[str, str] = {"language": language, "x_vector_only_mode": _form_value(x_vector_only_mode)}
            files: dict[str, tuple[str, Any, str]] = {}

            if ref_text is not None:
                data["ref_text"] = ref_text
            if output_prefix is not None:
                data["output_prefix"] = output_prefix
            for key, value in gen_kwargs.items():
                if value is not None:
                    data[key] = _form_value(value)

            ref_audio = Path(ref_audio_path).expanduser().resolve()
            files["ref_audio"] = (ref_audio.name, stack.enter_context(ref_audio.open("rb")), "audio/*")

            synthesis_text_file = Path(text_file).expanduser().resolve()
            files["text_file"] = (synthesis_text_file.name, stack.enter_context(synthesis_text_file.open("rb")), "text/plain")

            if ref_text_file is not None:
                path = Path(ref_text_file).expanduser().resolve()
                files["ref_text_file"] = (path.name, stack.enter_context(path.open("rb")), "text/plain")

            response = self._client.post(f"{API_PREFIX}/tts/voice_clone_batch_file", data=data, files=files)

        payload = self._json(response)
        if download_dir is not None:
            target_dir = Path(download_dir).expanduser().resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            for item in payload["audio_paths"]:
                self.download_url(item["url"], target_dir / item["filename"])
        return payload

    def narration_batch_file(
        self,
        *,
        text_file: str | Path,
        language: str = "Auto",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        output_prefix: Optional[str] = None,
        download_dir: Optional[str | Path] = None,
        **gen_kwargs,
    ) -> dict[str, Any]:
        with ExitStack() as stack:
            data: dict[str, str] = {"language": language}
            files: dict[str, tuple[str, Any, str]] = {}

            if speaker is not None:
                data["speaker"] = speaker
            if instruct is not None:
                data["instruct"] = instruct
            if output_prefix is not None:
                data["output_prefix"] = output_prefix
            for key, value in gen_kwargs.items():
                if value is not None:
                    data[key] = _form_value(value)

            synthesis_text_file = Path(text_file).expanduser().resolve()
            files["text_file"] = (synthesis_text_file.name, stack.enter_context(synthesis_text_file.open("rb")), "text/plain")

            response = self._client.post(f"{API_PREFIX}/tts/narration_batch_file", data=data, files=files)

        payload = self._json(response)
        if download_dir is not None:
            target_dir = Path(download_dir).expanduser().resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            for item in payload["audio_paths"]:
                self.download_url(item["url"], target_dir / item["filename"])
        return payload
