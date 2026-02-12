#!/usr/bin/env python3
"""
Visualize TTS streaming chunk boundaries from a tesseract debug dump.

Usage:
    python scripts/visualize_chunks.py /tmp/tesseract-debug/<timestamp>/

Reads metadata.json, full_stream.wav, and raw/processed chunk files.
Produces PNG plots in the same directory.

Dependencies: numpy, matplotlib, scipy
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram


def load_metadata(dump_dir: Path) -> dict:
    with open(dump_dir / "metadata.json") as f:
        return json.load(f)


def load_wav(dump_dir: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(dump_dir / "full_stream.wav")
    return sr, data.astype(np.float32)


def load_raw_chunks(dump_dir: Path, subfolder: str, count: int) -> list[np.ndarray]:
    chunks = []
    folder = dump_dir / subfolder
    for i in range(count):
        path = folder / f"chunk_{i:03d}.raw"
        if path.exists():
            chunks.append(np.fromfile(path, dtype=np.float32))
        else:
            chunks.append(np.array([], dtype=np.float32))
    return chunks


def plot_waveform_overview(audio: np.ndarray, sr: int, chunks: list[dict],
                           blend: int, dump_dir: Path):
    """Plot 1: Full waveform with chunk boundary markers and crossfade regions."""
    fig, ax = plt.subplots(figsize=(16, 4))
    t = np.arange(len(audio)) / sr
    ax.plot(t, audio, linewidth=0.3, color="#2196F3", alpha=0.8)

    for chunk in chunks:
        offset = chunk["scheduledOffset"]
        size = chunk["scheduledSize"]
        if size == 0:
            continue
        boundary = offset / sr
        ax.axvline(boundary, color="#F44336", alpha=0.5, linewidth=0.8, linestyle="--")

        # Highlight crossfade region (blend samples before each boundary)
        if offset > 0:
            xfade_start = max(0, offset - blend) / sr
            xfade_end = offset / sr
            ax.axvspan(xfade_start, xfade_end, alpha=0.15, color="#FF9800")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Full Stream Waveform with Chunk Boundaries")
    ax.set_xlim(0, len(audio) / sr)
    fig.tight_layout()
    fig.savefig(dump_dir / "01_waveform_overview.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 01_waveform_overview.png")


def plot_boundary_zooms(audio: np.ndarray, sr: int, chunks: list[dict],
                        blend: int, dump_dir: Path):
    """Plot 2: Zoomed view around each chunk boundary."""
    # Collect boundaries (skip first chunk which has no predecessor)
    boundaries = []
    for chunk in chunks:
        if chunk["scheduledOffset"] > 0 and chunk["scheduledSize"] > 0:
            boundaries.append(chunk["scheduledOffset"])

    if not boundaries:
        print("  No boundaries to zoom into.")
        return

    window_samples = int(0.1 * sr)  # 100ms window

    cols = min(3, len(boundaries))
    rows = (len(boundaries) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), squeeze=False)

    for idx, boundary in enumerate(boundaries):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        start = max(0, boundary - window_samples)
        end = min(len(audio), boundary + window_samples)
        segment = audio[start:end]
        t = np.arange(start, end) / sr

        ax.plot(t, segment, linewidth=0.5, color="#2196F3")
        ax.axvline(boundary / sr, color="#F44336", linewidth=1.5, label="boundary")

        # Mark crossfade region
        xfade_start = max(0, boundary - blend) / sr
        xfade_end = boundary / sr
        ax.axvspan(xfade_start, xfade_end, alpha=0.2, color="#FF9800", label="crossfade")

        # Amplitude envelope
        env_window = min(64, len(segment) // 4)
        if env_window > 0 and len(segment) > env_window:
            envelope = np.convolve(np.abs(segment), np.ones(env_window) / env_window, mode="same")
            ax.plot(t, envelope, color="#4CAF50", linewidth=1, alpha=0.7, label="envelope")

        ax.set_title(f"Boundary {idx} @ {boundary / sr:.4f}s", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(len(boundaries), rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Zoomed Boundary Views (~100ms window)", fontsize=12)
    fig.tight_layout()
    fig.savefig(dump_dir / "02_boundary_zooms.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 02_boundary_zooms.png")


def plot_spectrogram(audio: np.ndarray, sr: int, chunks: list[dict],
                     dump_dir: Path):
    """Plot 3: Spectrogram with boundary lines overlaid."""
    fig, ax = plt.subplots(figsize=(16, 5))

    nperseg = min(1024, len(audio) // 4) if len(audio) > 256 else len(audio)
    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)

    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="magma")

    for chunk in chunks:
        offset = chunk["scheduledOffset"]
        if offset > 0 and chunk["scheduledSize"] > 0:
            ax.axvline(offset / sr, color="cyan", alpha=0.6, linewidth=0.8, linestyle="--")

    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram with Chunk Boundaries")
    ax.set_ylim(0, min(8000, sr // 2))
    fig.tight_layout()
    fig.savefig(dump_dir / "03_spectrogram.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 03_spectrogram.png")


def plot_discontinuity_metrics(audio: np.ndarray, sr: int, chunks: list[dict],
                               blend: int, dump_dir: Path):
    """Plot 4: Per-boundary discontinuity metrics."""
    boundaries = []
    for chunk in chunks:
        if chunk["scheduledOffset"] > 0 and chunk["scheduledSize"] > 0:
            boundaries.append(chunk["scheduledOffset"])

    if not boundaries:
        print("  No boundaries for discontinuity metrics.")
        return

    max_deltas = []
    rms_values = []
    zcr_changes = []

    analysis_window = min(256, blend * 2)

    for boundary in boundaries:
        # Max sample delta at boundary
        if 0 < boundary < len(audio):
            delta = abs(float(audio[boundary]) - float(audio[boundary - 1]))
        else:
            delta = 0.0
        max_deltas.append(delta)

        # RMS in window around boundary
        start = max(0, boundary - analysis_window)
        end = min(len(audio), boundary + analysis_window)
        segment = audio[start:end]
        rms = float(np.sqrt(np.mean(segment ** 2))) if len(segment) > 0 else 0.0
        rms_values.append(rms)

        # Zero-crossing rate change (before vs after boundary)
        half = analysis_window
        before = audio[max(0, boundary - half):boundary]
        after = audio[boundary:min(len(audio), boundary + half)]

        def zcr(x):
            if len(x) < 2:
                return 0.0
            return float(np.sum(np.abs(np.diff(np.sign(x))) > 0)) / len(x)

        zcr_before = zcr(before)
        zcr_after = zcr(after)
        zcr_changes.append(abs(zcr_after - zcr_before))

    x = np.arange(len(boundaries))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].bar(x, max_deltas, color="#F44336", alpha=0.8)
    axes[0].set_ylabel("Max Sample Delta")
    axes[0].set_title("Discontinuity at Each Boundary")

    axes[1].bar(x, rms_values, color="#FF9800", alpha=0.8)
    axes[1].set_ylabel("RMS (±window)")

    axes[2].bar(x, zcr_changes, color="#9C27B0", alpha=0.8)
    axes[2].set_ylabel("ZCR Change")
    axes[2].set_xlabel("Boundary Index")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b / sr:.3f}s" for b in boundaries], rotation=45, fontsize=7)

    fig.suptitle("Discontinuity Metrics per Chunk Boundary", fontsize=12)
    fig.tight_layout()
    fig.savefig(dump_dir / "04_discontinuity_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 04_discontinuity_metrics.png")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <debug-dump-directory>")
        sys.exit(1)

    dump_dir = Path(sys.argv[1])
    if not dump_dir.is_dir():
        print(f"Error: {dump_dir} is not a directory")
        sys.exit(1)

    print(f"Loading debug dump from {dump_dir}")

    meta = load_metadata(dump_dir)
    sr, audio = load_wav(dump_dir)
    chunks = meta["chunks"]
    blend = meta["blendSamples"]

    print(f"  Sample rate: {sr} Hz")
    print(f"  Blend samples: {blend} ({blend / sr * 1000:.1f} ms)")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Total audio: {len(audio)} samples ({len(audio) / sr:.2f}s)")
    print()

    # Load raw chunks for reference
    raw_chunks = load_raw_chunks(dump_dir, "raw_chunks", len(chunks))
    processed_chunks = load_raw_chunks(dump_dir, "processed_chunks", len(chunks))

    print("Generating plots...")
    plot_waveform_overview(audio, sr, chunks, blend, dump_dir)
    plot_boundary_zooms(audio, sr, chunks, blend, dump_dir)
    plot_spectrogram(audio, sr, chunks, dump_dir)
    plot_discontinuity_metrics(audio, sr, chunks, blend, dump_dir)

    print(f"\nAll plots saved to {dump_dir}")


if __name__ == "__main__":
    main()
