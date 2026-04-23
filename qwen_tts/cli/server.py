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

from dataclasses import replace
from pathlib import Path

import uvicorn

from .fastapi_service import ServiceSettings, build_parser, create_app


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = ServiceSettings.from_env()
    if args.storage_root:
        settings = replace(settings, storage_root=Path(args.storage_root).expanduser().resolve())
    if args.base_model:
        settings = replace(settings, base_model=args.base_model)
    if args.custom_model:
        settings = replace(settings, custom_model=args.custom_model)
    if args.voice_design_model:
        settings = replace(settings, voice_design_model=args.voice_design_model)
    if args.device:
        settings = replace(settings, device=args.device)
    if args.dtype:
        settings = replace(settings, dtype=args.dtype)
    if args.attn_implementation:
        settings = replace(settings, attn_implementation=args.attn_implementation)
    if args.narrator_speaker:
        settings = replace(settings, narrator_speaker=args.narrator_speaker)

    app = create_app(settings=settings)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
