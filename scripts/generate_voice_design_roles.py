from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = ROOT / "outputs" / "voice_design_roles"


ROLES = [
    {
        "role": "Voice-over",
        "text": "その夜、静かな街に小さな変化が訪れました。誰も気づかないうちに、物語はゆっくりと動き始めていたのです。",
        "instruct": (
            "Japanese male voice, natural story narration style for reading plot aloud, clear standard Japanese, "
            "ordinary adult male tone, calm and slightly warm, audiobook-like pacing, not an announcer, "
            "not a training coach, with only a very subtle clean electronic texture."
        ),
    },
    {
        "role": "Trainer",
        "text": "いいですね。その姿勢を保って、次はゆっくりと腕を上げてください。無理をせず、自分のペースで続けましょう。",
        "instruct": (
            "Japanese male voice, friendly trainer and instructor style, clear standard Japanese, "
            "ordinary adult male tone, lightly energetic but not exaggerated, with a subtle electronic assistant feel."
        ),
    },
]


def choose_runtime() -> dict:
    if torch.cuda.is_available():
        return {
            "device_map": "cuda:0",
            "dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        }
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {
            "device_map": "mps",
            "dtype": torch.float16,
            "attn_implementation": "sdpa",
        }
    return {
        "device_map": "cpu",
        "dtype": torch.float32,
        "attn_implementation": "sdpa",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roles", nargs="*", help="Optional role names to generate, such as Voice-over or Trainer.")
    args = parser.parse_args()
    selected = set(args.roles)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = choose_runtime()
    print(f"Loading VoiceDesign model from {MODEL_DIR}")
    print(f"Runtime: {runtime}")
    model = Qwen3TTSModel.from_pretrained(str(MODEL_DIR), **runtime)

    for item in ROLES:
        if selected and item["role"] not in selected:
            continue
        role_dir = OUTPUT_DIR / item["role"]
        role_dir.mkdir(parents=True, exist_ok=True)
        (role_dir / "text.txt").write_text(item["text"] + "\n", encoding="utf-8")
        (role_dir / "instruct.txt").write_text(item["instruct"] + "\n", encoding="utf-8")

        print(f"Generating {item['role']}...")
        wavs, sr = model.generate_voice_design(
            text=item["text"],
            language="Japanese",
            instruct=item["instruct"],
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
        )
        sf.write(role_dir / "reference.wav", wavs[0], sr)
        print(f"Wrote {role_dir / 'reference.wav'} ({sr} Hz)")


if __name__ == "__main__":
    main()
