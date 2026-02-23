"""Filterbank generation for Mel spectrograms."""

from typing import Literal

import numpy as np
import numpy.typing as npt


def _hz_to_mel(
    freq: float | npt.NDArray[np.float64], mel_scale: Literal["htk", "kaldi", "slaney"]
) -> npt.NDArray[np.float64]:
    if mel_scale == "htk":
        return 2595 * np.log10(1.0 + freq / 700.0)
    if mel_scale == "kaldi":
        return 1127 * np.log(1.0 + freq / 700.0)
    return np.where(
        freq < 1000, 3 * freq / 200.0, 15 + 27 * np.log(freq / 1000.0 + np.finfo(np.float32).eps) / np.log(6.4)
    )


def _mel_to_hz(mels: npt.NDArray[np.float64], mel_scale: Literal["htk", "slaney"]) -> npt.NDArray[np.float64]:
    if mel_scale == "htk":
        return 700 * (np.pow(10.0, mels / 2595.0) - 1.0)
    return np.where(mels < 15, 200 * mels / 3.0, 1000 * np.pow(6.4, ((mels - 15) / 27.0)))


def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Literal["slaney"] | None = None,
    mel_scale: Literal["htk", "slaney", "kaldi"] = "htk",
) -> npt.NDArray[np.float64]:
    if f_max <= 0.0:
        f_max += sample_rate / 2

    all_freqs = np.linspace(0, sample_rate // 2, n_freqs)
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    if mel_scale == "kaldi":
        mel = _hz_to_mel(all_freqs, mel_scale=mel_scale)
    else:
        mel = all_freqs
        m_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    up_slopes = (mel[:, None] - m_pts[:-2]) / (m_pts[1:-1] - m_pts[:-2])
    down_slopes = (m_pts[2:] - mel[:, None]) / (m_pts[2:] - m_pts[1:-1])
    fb = np.maximum(0.0, np.minimum(up_slopes, down_slopes))

    if norm == "slaney":
        fb *= 2.0 / (m_pts[2:] - m_pts[:-2])

    return fb
