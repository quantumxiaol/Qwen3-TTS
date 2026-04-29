from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REF_AUDIO_PATH = REPO_ROOT / "ref_audio" / "reference.mp3"
DEFAULT_REF_TEXT_PATH = REPO_ROOT / "ref_audio" / "reference_jp.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "voice_clone"

SYNTHESIS_CASES = [
    (
        "zh",
        "Chinese",
        "你好，我是这次的语音克隆测试，很高兴认识你。",
    ),
    (
        "ja",
        "Japanese",
        "こんにちは、今回の音声クローンのテストです。お会いできてうれしいです。",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen3-TTS voice clone smoke test.")
    parser.add_argument(
        "--case",
        choices=("all", "zh", "ja"),
        default=os.environ.get("QWEN3_TTS_CASE", "all"),
        help="Synthesis case to run: all, zh, or ja. Can also be set with QWEN3_TTS_CASE.",
    )
    return parser.parse_args()


def resolve_path(env_name: str, default_path: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else default_path


def parse_bool_env(env_name: str, default: bool) -> bool:
    value = os.environ.get(env_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def synchronize_device(device_name: str) -> None:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    if device_name.startswith("mps") and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def choose_runtime() -> tuple[dict, str]:
    device_override = os.environ.get("QWEN3_TTS_DEVICE")
    if device_override:
        device_name = device_override
    elif torch.cuda.is_available():
        device_name = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    if device_name.startswith("cuda"):
        runtime = {
            "device_map": device_name,
            "dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        }
    elif device_name.startswith("mps"):
        runtime = {
            "device_map": device_name,
            "dtype": torch.float16,
            "attn_implementation": "sdpa",
        }
    else:
        runtime = {
            "device_map": device_name,
            "dtype": torch.float32,
            "attn_implementation": "sdpa",
        }
    return runtime, device_name


def load_reference_text(ref_text_path: Path) -> str:
    ref_text = ref_text_path.read_text(encoding="utf-8").strip()
    if not ref_text:
        raise ValueError(f"Reference transcript is empty: {ref_text_path}")
    return ref_text


def now() -> float:
    return time.perf_counter()


def select_synthesis_cases(case: str) -> list[tuple[str, str, str]]:
    if case == "all":
        return SYNTHESIS_CASES
    selected = [item for item in SYNTHESIS_CASES if item[0] == case]
    if not selected:
        raise ValueError(f"No synthesis case matched: {case}")
    return selected


def main() -> None:
    args = parse_args()
    synthesis_cases = select_synthesis_cases(args.case)

    model_path = resolve_path("QWEN3_TTS_BASE_MODEL", DEFAULT_MODEL_PATH)
    ref_audio_path = resolve_path("QWEN3_TTS_REF_AUDIO", DEFAULT_REF_AUDIO_PATH)
    ref_text_path = resolve_path("QWEN3_TTS_REF_TEXT", DEFAULT_REF_TEXT_PATH)
    output_dir = resolve_path("QWEN3_TTS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

    for path in [model_path, ref_audio_path, ref_text_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ref_text = load_reference_text(ref_text_path)
    runtime_kwargs, device_name = choose_runtime()

    print(f"[runtime] device={device_name} dtype={runtime_kwargs['dtype']} attn={runtime_kwargs['attn_implementation']}")
    print(f"[paths] model={model_path}")
    print(f"[paths] ref_audio={ref_audio_path}")
    print(f"[paths] ref_text={ref_text_path}")
    print(f"[paths] output_dir={output_dir}")
    print(f"[case] selected={args.case} count={len(synthesis_cases)}")

    t0 = now()
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        **runtime_kwargs,
    )
    synchronize_device(device_name)
    t1 = now()

    t2 = now()
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    synchronize_device(device_name)
    t3 = now()

    texts = [item[2] for item in synthesis_cases]
    languages = [item[1] for item in synthesis_cases]
    do_sample = parse_bool_env("QWEN3_TTS_DO_SAMPLE", False)

    t4 = now()
    wavs, sr = model.generate_voice_clone(
        text=texts,
        language=languages,
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=2048,
        do_sample=do_sample,
        top_k=50,
        top_p=1.0,
        temperature=0.9 if do_sample else 1.0,
        repetition_penalty=1.05,
        subtalker_dosample=do_sample,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9 if do_sample else 1.0,
    )
    synchronize_device(device_name)
    t5 = now()

    stem = ref_audio_path.stem.replace(" ", "_").lower()
    for (suffix, language, text), wav in zip(synthesis_cases, wavs):
        output_path = output_dir / f"{stem}_{suffix}.wav"
        sf.write(output_path, wav, sr)
        print(f"[saved] {output_path} | language={language} | text={text}")

    audio_durations = [len(wav) / sr for wav in wavs]
    total_audio_duration = sum(audio_durations)
    generation_time = t5 - t4
    rtf = generation_time / total_audio_duration if total_audio_duration > 0 else float("inf")

    print(f"[timing] load_model={t1 - t0:.3f}s")
    print(f"[timing] build_prompt={t3 - t2:.3f}s")
    print(f"[timing] generate={generation_time:.3f}s")
    print(f"[timing] total_audio={total_audio_duration:.3f}s")
    print(f"[timing] rtf={rtf:.3f}")
    for (suffix, language, _), duration in zip(synthesis_cases, audio_durations):
        print(f"[timing] sample={suffix} language={language} audio_duration={duration:.3f}s")


if __name__ == "__main__":
    main()
