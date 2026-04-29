from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


DEFAULT_SILENCE_DBFS = -45.0
FRAME_MS = 25.0
HOP_MS = 10.0


def format_seconds(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    minutes, seconds = divmod(value, 60.0)
    if minutes >= 1:
        return f"{int(minutes)}m{seconds:05.2f}s"
    return f"{seconds:.3f}s"


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def ffprobe_info(path: Path) -> dict[str, Any] | None:
    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,codec_long_name,sample_rate,channels,channel_layout,duration,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    return streams[0] if streams else None


def soundfile_info(path: Path) -> dict[str, Any] | None:
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return None
    return {
        "samplerate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.duration,
        "format": info.format,
        "subtype": info.subtype,
    }


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(path), sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    return np.asarray(audio, dtype=np.float32), int(sr)


def frame_rms(mono: np.ndarray, sr: int) -> np.ndarray:
    frame_length = max(1, int(sr * FRAME_MS / 1000.0))
    hop_length = max(1, int(sr * HOP_MS / 1000.0))
    if mono.size == 0:
        return np.array([], dtype=np.float32)
    return librosa.feature.rms(
        y=mono,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]


def edge_silence_seconds(rms: np.ndarray, sr: int, threshold: float) -> tuple[float, float]:
    if rms.size == 0:
        return 0.0, 0.0

    hop_seconds = max(1, int(sr * HOP_MS / 1000.0)) / sr
    quiet = rms < threshold

    leading_frames = 0
    for is_quiet in quiet:
        if not is_quiet:
            break
        leading_frames += 1

    trailing_frames = 0
    for is_quiet in quiet[::-1]:
        if not is_quiet:
            break
        trailing_frames += 1

    return leading_frames * hop_seconds, trailing_frames * hop_seconds


def audio_stats(audio: np.ndarray, sr: int, silence_dbfs: float) -> dict[str, Any]:
    samples = audio.shape[-1]
    duration = samples / sr if sr > 0 else 0.0
    mono = np.mean(audio, axis=0)

    finite = np.isfinite(audio)
    peak = float(np.max(np.abs(audio[finite]))) if np.any(finite) else float("nan")
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    mean = float(np.mean(mono)) if mono.size else 0.0
    mean_abs = float(np.mean(np.abs(mono))) if mono.size else 0.0

    channel_rms = np.sqrt(np.mean(np.square(audio), axis=1)) if samples else np.zeros(audio.shape[0])
    channel_peak = np.max(np.abs(audio), axis=1) if samples else np.zeros(audio.shape[0])

    rms_frames = frame_rms(mono, sr)
    silence_threshold = 10.0 ** (silence_dbfs / 20.0)
    quiet_ratio = float(np.mean(rms_frames < silence_threshold)) if rms_frames.size else 0.0
    leading_silence, trailing_silence = edge_silence_seconds(rms_frames, sr, silence_threshold)

    return {
        "channels": int(audio.shape[0]),
        "sample_rate": sr,
        "samples_per_channel": int(samples),
        "duration": duration,
        "peak": peak,
        "peak_dbfs": dbfs(peak),
        "rms": rms,
        "rms_dbfs": dbfs(rms),
        "mean": mean,
        "mean_abs": mean_abs,
        "non_finite_samples": int(np.size(audio) - np.count_nonzero(finite)),
        "clipped_sample_ratio": float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0,
        "quiet_frame_ratio": quiet_ratio,
        "leading_silence": leading_silence,
        "trailing_silence": trailing_silence,
        "channel_rms": [float(x) for x in channel_rms],
        "channel_peak": [float(x) for x in channel_peak],
    }


def build_warnings(stats: dict[str, Any], text: str | None) -> list[str]:
    warnings: list[str] = []
    duration = float(stats["duration"])
    peak = float(stats["peak"])
    rms_db = float(stats["rms_dbfs"])
    quiet_ratio = float(stats["quiet_frame_ratio"])
    leading = float(stats["leading_silence"])
    trailing = float(stats["trailing_silence"])

    if duration < 2.0:
        warnings.append("audio is very short for voice cloning (<2s)")
    if duration > 30.0:
        warnings.append("audio is long for a reference prompt (>30s); consider trimming")
    if not math.isfinite(peak) or stats["non_finite_samples"]:
        warnings.append("audio contains non-finite samples")
    if peak < 0.02:
        warnings.append("audio peak is very low")
    if rms_db < -35.0:
        warnings.append("audio RMS level is low")
    if quiet_ratio > 0.5:
        warnings.append("more than half of analysis frames are below the silence threshold")
    if leading > 0.5:
        warnings.append("leading silence is longer than 0.5s")
    if trailing > 0.5:
        warnings.append("trailing silence is longer than 0.5s")
    if stats["clipped_sample_ratio"] > 0.001:
        warnings.append("possible clipping: many samples are near full scale")

    if text is not None:
        text_len = len("".join(text.split()))
        if text_len == 0:
            warnings.append("reference text is empty")
        elif duration > 0:
            chars_per_second = text_len / duration
            if chars_per_second < 0.8:
                warnings.append("reference text is short relative to audio duration")
            if chars_per_second > 12.0:
                warnings.append("reference text is long relative to audio duration")

    return warnings


def print_report(path: Path, text_path: Path | None, silence_dbfs: float) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file does not exist: {resolved}")

    print(f"[file] path={resolved}")
    print(f"[file] size_bytes={resolved.stat().st_size}")

    sf_meta = soundfile_info(resolved)
    if sf_meta is None:
        print("[container:soundfile] unavailable")
    else:
        print(
            "[container:soundfile] "
            f"format={sf_meta['format']} subtype={sf_meta['subtype']} "
            f"channels={sf_meta['channels']} sample_rate={sf_meta['samplerate']} "
            f"frames={sf_meta['frames']} duration={format_seconds(float(sf_meta['duration']))}"
        )

    ff_meta = ffprobe_info(resolved)
    if ff_meta is not None:
        print(
            "[container:ffprobe] "
            f"codec={ff_meta.get('codec_name', 'n/a')} "
            f"sample_rate={ff_meta.get('sample_rate', 'n/a')} "
            f"channels={ff_meta.get('channels', 'n/a')} "
            f"duration={ff_meta.get('duration', 'n/a')} "
            f"bit_rate={ff_meta.get('bit_rate', 'n/a')}"
        )

    audio, sr = load_audio(resolved)
    stats = audio_stats(audio, sr, silence_dbfs)

    print(
        "[decoded] "
        f"channels={stats['channels']} sample_rate={stats['sample_rate']} "
        f"samples_per_channel={stats['samples_per_channel']} "
        f"duration={format_seconds(float(stats['duration']))}"
    )
    print(
        "[level] "
        f"peak={stats['peak']:.6f} peak_dbfs={stats['peak_dbfs']:.2f} "
        f"rms={stats['rms']:.6f} rms_dbfs={stats['rms_dbfs']:.2f} "
        f"mean={stats['mean']:.6f} mean_abs={stats['mean_abs']:.6f}"
    )
    print(
        "[silence] "
        f"threshold_dbfs={silence_dbfs:.1f} "
        f"quiet_frames={stats['quiet_frame_ratio'] * 100.0:.1f}% "
        f"leading={format_seconds(float(stats['leading_silence']))} "
        f"trailing={format_seconds(float(stats['trailing_silence']))}"
    )
    print(
        "[quality] "
        f"non_finite_samples={stats['non_finite_samples']} "
        f"clipped_sample_ratio={stats['clipped_sample_ratio'] * 100.0:.4f}%"
    )

    for idx, (peak, rms) in enumerate(zip(stats["channel_peak"], stats["channel_rms"]), start=1):
        print(f"[channel:{idx}] peak={peak:.6f} rms={rms:.6f} rms_dbfs={dbfs(rms):.2f}")

    text = None
    if text_path is not None:
        resolved_text = text_path.expanduser().resolve()
        text = read_text(resolved_text)
        text_len = len("".join(text.split()))
        chars_per_second = text_len / stats["duration"] if stats["duration"] else float("inf")
        print(f"[text] path={resolved_text}")
        print(f"[text] chars_no_space={text_len} chars_per_second={chars_per_second:.3f}")

    warnings = build_warnings(stats, text)
    if warnings:
        for warning in warnings:
            print(f"[warning] {warning}")
    else:
        print("[warning] none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an audio file for Qwen3-TTS reference use.")
    parser.add_argument("audio", type=Path, help="Path to an audio file.")
    parser.add_argument("--text-file", type=Path, help="Optional transcript/reference text file.")
    parser.add_argument(
        "--silence-dbfs",
        type=float,
        default=DEFAULT_SILENCE_DBFS,
        help=f"Frame RMS threshold for silence checks. Default: {DEFAULT_SILENCE_DBFS}.",
    )
    args = parser.parse_args()

    print_report(args.audio, args.text_file, args.silence_dbfs)


if __name__ == "__main__":
    main()
