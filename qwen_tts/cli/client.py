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
import json
from pathlib import Path
from typing import Any

from ..httpx_client import DEFAULT_SERVER_URL, Qwen3TTSHttpxClient


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--do-sample", default=None, choices=["true", "false"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--subtalker-top-k", type=int, default=None)
    parser.add_argument("--subtalker-top-p", type=float, default=None)
    parser.add_argument("--subtalker-temperature", type=float, default=None)


def _collect_gen_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    do_sample = None
    if args.do_sample == "true":
        do_sample = True
    elif args.do_sample == "false":
        do_sample = False

    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-client",
        description="HTTP client for the Qwen3-TTS FastAPI service.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"Server origin without API prefix (default: {DEFAULT_SERVER_URL}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Query service health.")
    subparsers.add_parser("narrators", help="List supported narration speakers.")

    download_parser = subparsers.add_parser("download", help="Download a file by URL.")
    download_parser.add_argument("--url", required=True, help="Absolute file URL returned by the service.")
    download_parser.add_argument("--output", required=True, help="Local output path.")

    voice_clone_parser = subparsers.add_parser("voice-clone", help="Call /qwen3tts/tts/voice_clone.")
    voice_clone_parser.add_argument("--ref-audio", required=True, help="Reference audio path.")
    voice_clone_parser.add_argument("--text", default=None, help="Synthesis text.")
    voice_clone_parser.add_argument("--text-file", default=None, help="Synthesis text file path.")
    voice_clone_parser.add_argument("--ref-text", default=None, help="Reference transcript.")
    voice_clone_parser.add_argument("--ref-text-file", default=None, help="Reference transcript file path.")
    voice_clone_parser.add_argument("--language", default="Auto", help="Synthesis language.")
    voice_clone_parser.add_argument("--output-name", default=None, help="Requested output filename.")
    voice_clone_parser.add_argument("--x-vector-only-mode", action="store_true", help="Use speaker embedding only.")
    voice_clone_parser.add_argument("--download-to", default=None, help="Optional local wav path to download after synthesis.")
    _add_generation_args(voice_clone_parser)

    voice_clone_batch_parser = subparsers.add_parser("voice-clone-batch-file", help="Call /qwen3tts/tts/voice_clone_batch_file.")
    voice_clone_batch_parser.add_argument("--ref-audio", required=True, help="Reference audio path.")
    voice_clone_batch_parser.add_argument("--text-file", required=True, help="Batch synthesis text file path.")
    voice_clone_batch_parser.add_argument("--ref-text", default=None, help="Reference transcript.")
    voice_clone_batch_parser.add_argument("--ref-text-file", default=None, help="Reference transcript file path.")
    voice_clone_batch_parser.add_argument("--language", default="Auto", help="Synthesis language.")
    voice_clone_batch_parser.add_argument("--output-prefix", default=None, help="Prefix for generated wav names.")
    voice_clone_batch_parser.add_argument("--x-vector-only-mode", action="store_true", help="Use speaker embedding only.")
    voice_clone_batch_parser.add_argument("--download-dir", default=None, help="Optional local directory to download generated wavs.")
    _add_generation_args(voice_clone_batch_parser)

    narration_parser = subparsers.add_parser("narration", help="Call /qwen3tts/tts/narration.")
    narration_parser.add_argument("--text", default=None, help="Narration text.")
    narration_parser.add_argument("--text-file", default=None, help="Narration text file path.")
    narration_parser.add_argument("--language", default="Auto", help="Narration language.")
    narration_parser.add_argument("--speaker", default=None, help="CustomVoice speaker name.")
    narration_parser.add_argument("--instruct", default=None, help="Optional instruction text.")
    narration_parser.add_argument("--output-name", default=None, help="Requested output filename.")
    narration_parser.add_argument("--download-to", default=None, help="Optional local wav path to download after synthesis.")
    _add_generation_args(narration_parser)

    narration_batch_parser = subparsers.add_parser("narration-batch-file", help="Call /qwen3tts/tts/narration_batch_file.")
    narration_batch_parser.add_argument("--text-file", required=True, help="Batch narration text file path.")
    narration_batch_parser.add_argument("--language", default="Auto", help="Narration language.")
    narration_batch_parser.add_argument("--speaker", default=None, help="CustomVoice speaker name.")
    narration_batch_parser.add_argument("--instruct", default=None, help="Optional instruction text.")
    narration_batch_parser.add_argument("--output-prefix", default=None, help="Prefix for generated wav names.")
    narration_batch_parser.add_argument("--download-dir", default=None, help="Optional local directory to download generated wavs.")
    _add_generation_args(narration_batch_parser)

    return parser


def _print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    with Qwen3TTSHttpxClient(server_url=args.server_url) as client:
        if args.command == "health":
            _print_payload(client.health())
            return

        if args.command == "narrators":
            _print_payload(client.list_narrators())
            return

        if args.command == "download":
            target = client.download_url(args.url, args.output)
            print(json.dumps({"status": "success", "output_path": str(Path(target))}, ensure_ascii=False, indent=2))
            return

        gen_kwargs = _collect_gen_kwargs(args)
        if args.command == "voice-clone":
            _print_payload(
                client.voice_clone(
                    ref_audio_path=args.ref_audio,
                    text=args.text,
                    text_file=args.text_file,
                    ref_text=args.ref_text,
                    ref_text_file=args.ref_text_file,
                    language=args.language,
                    output_name=args.output_name,
                    x_vector_only_mode=args.x_vector_only_mode,
                    download_to=args.download_to,
                    **gen_kwargs,
                )
            )
            return

        if args.command == "voice-clone-batch-file":
            _print_payload(
                client.voice_clone_batch_file(
                    ref_audio_path=args.ref_audio,
                    text_file=args.text_file,
                    ref_text=args.ref_text,
                    ref_text_file=args.ref_text_file,
                    language=args.language,
                    output_prefix=args.output_prefix,
                    x_vector_only_mode=args.x_vector_only_mode,
                    download_dir=args.download_dir,
                    **gen_kwargs,
                )
            )
            return

        if args.command == "narration":
            _print_payload(
                client.narration(
                    text=args.text,
                    text_file=args.text_file,
                    language=args.language,
                    speaker=args.speaker,
                    instruct=args.instruct,
                    output_name=args.output_name,
                    download_to=args.download_to,
                    **gen_kwargs,
                )
            )
            return

        if args.command == "narration-batch-file":
            _print_payload(
                client.narration_batch_file(
                    text_file=args.text_file,
                    language=args.language,
                    speaker=args.speaker,
                    instruct=args.instruct,
                    output_prefix=args.output_prefix,
                    download_dir=args.download_dir,
                    **gen_kwargs,
                )
            )
            return

        raise SystemExit(f"Unsupported command: {args.command}")
