from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import librosa
import numpy as np
import pandas as pd
import parselmouth
import scipy.signal as sg
import webrtcvad
from scipy.signal import find_peaks, hilbert
from scipy.spatial import ConvexHull
from scipy.stats import median_abs_deviation, theilslopes

# Опционально: crepe может отсутствовать в окружении
try:
    import crepe
except ModuleNotFoundError:
    crepe = None

from helpers.base_extractor import BaseExtractor

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated as an API", category=UserWarning
)


# ---------------------------------------------------------------------------
# Общие типы и константы
# ---------------------------------------------------------------------------
FrameSpec = Tuple[int, int]  # (frame_length, hop_length) в сэмплах
EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Вспомогательные функции (глобальные, чтобы кешировать и делить между блоками)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def ms_to_samples(ms: float, sr: int) -> int:
    """Миллисекунды → сэмплы с кэшированием."""
    return int(round(ms * sr / 1000))


def safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def safe_std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if arr.size else 0.0


def robust_median(arr: np.ndarray) -> float:
    return float(np.median(arr)) if arr.size else 0.0


def safe_max(arr: np.ndarray) -> float:
    return float(np.max(arr)) if arr.size else 0.0


def remove_outliers(arr: np.ndarray, m: float = 3.5) -> np.ndarray:
    if arr.size == 0:
        return arr
    med = np.median(arr)
    mad = median_abs_deviation(arr) + EPS
    return arr[np.abs(arr - med) / mad < m]


def hz_to_bark(f: np.ndarray | float) -> np.ndarray | float:
    f_arr = np.asarray(f)
    return 6 * np.arcsinh(f_arr / 600.0)


# ---------------------------------------------------------------------------
# Основной класс извлечения признаков
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FeatureExtractor:
    """
    Извлекает широкий набор акустических признаков русской речи.
    """

    use_vad: bool = True

    # ---------------------- публичные методы -----------------------
    def extract_from_file(self, wav_path: str | Path) -> pd.DataFrame:
        """Читает аудио и возвращает таблицу признаков по всему сигналу."""
        y, sr = librosa.load(str(wav_path), sr=16_000, mono=True)
        return self.extract_from_array(y, sr)

    def extract_from_array(self, y: np.ndarray, sr: int) -> pd.DataFrame:  # noqa: C901
        """Главный вход: numpy-сигнал + sr → DataFrame с признаками."""

        y = self._highpass_hz(y, sr)
        frame_len = ms_to_samples(25, sr)
        hop_len = ms_to_samples(5, sr)
        frame_spec: FrameSpec = (frame_len, hop_len)

        vad_mask = self._compute_vad_mask(y, sr, frame_spec) if self.use_vad else None

        feats: Dict[str, float] = {}
        feats.update(self._prosody(y, sr, hop_len, vad_mask))
        feats.update(self._voice(y, sr, frame_spec, vad_mask))
        feats.update(self._formants(y, sr, vad_mask))
        feats.update(self._spectral(y, sr, frame_spec))
        feats.update(self._articulation(y, sr, frame_spec))
        feats.update(self._dynamics(y, sr, frame_spec, vad_mask))

        return self._format_df(feats)

    # --------------------- обработка сигнала -----------------------
    def _highpass_hz(
        self,
        y: np.ndarray,
        sr: int,
        cutoff: float = 20.0,
        order: int = 1,
    ) -> np.ndarray:
        if order <= 0 or cutoff <= 0:
            return y
        sos = sg.butter(order, cutoff, fs=sr, btype="highpass", output="sos")
        return sg.sosfiltfilt(sos, y)

    def _format_df(self, feats: Dict[str, float]) -> pd.DataFrame:
        feats = {k: [v] for k, v in feats.items()}
        return pd.DataFrame(feats)

    # --------------------------- VAD -------------------------------
    def _compute_vad_mask(
        self,
        y: np.ndarray,
        sr: int,
        frame_spec: FrameSpec,
    ) -> np.ndarray | None:
        frame_len, hop_len = frame_spec
        if webrtcvad and sr in (8000, 16000, 32000, 48000):
            vad = webrtcvad.Vad(2)
            win_ms = 30
            win_len = int(sr * win_ms / 1000)
            hop_native = win_len
            n_frames = int(np.ceil(len(y) / hop_native))
            mask_native = np.zeros(n_frames, dtype=bool)
            for i in range(n_frames):
                segment = y[i * hop_native : i * hop_native + win_len]
                if segment.size < win_len:
                    segment = np.pad(segment, (0, win_len - segment.size))
                pcm16 = (segment * 32767).astype(np.int16).tobytes()
                mask_native[i] = vad.is_speech(pcm16, sr)
            factor = int(np.ceil(hop_native / hop_len))
            mask = np.repeat(mask_native, factor)[: int(np.ceil(len(y) / hop_len))]
        else:
            rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[
                0
            ]
            thr = np.percentile(rms, 5)
            mask = rms > thr
            mask = np.logical_or(mask, np.roll(mask, 1))
            mask = np.logical_or(mask, np.roll(mask, -1))
        return mask

    # --------------- Просодические признаки -----------------------
    def _prosody(
        self,
        y: np.ndarray,
        sr: int,
        hop_len: int,
        vad_mask: np.ndarray | None,
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        total_dur = len(y) / sr + EPS

        if vad_mask is None:
            rms = librosa.feature.rms(
                y=y, frame_length=ms_to_samples(20, sr), hop_length=hop_len
            )[0]
            thr = np.percentile(rms, 5)
            speech_mask = rms > thr
        else:
            speech_mask = vad_mask
        speech_time = np.sum(speech_mask) * hop_len / sr
        pause_mask = ~speech_mask

        # сгруппируем паузы/речевые отрезки
        pause_dur, speech_dur = [], []
        cur, in_pause = 0, bool(pause_mask[0])
        for flag in pause_mask:
            if flag == in_pause:
                cur += 1
            else:
                dur = cur * hop_len / sr
                (pause_dur if in_pause else speech_dur).append(dur)
                in_pause = flag
                cur = 1
        dur = cur * hop_len / sr
        (pause_dur if in_pause else speech_dur).append(dur)
        pause_dur = np.asarray(pause_dur)
        speech_dur = np.asarray(speech_dur)
        total_pause = safe_mean(pause_dur) * len(pause_dur) if pause_dur.size else 0

        feats.update(
            {
                "voiced_fraction": speech_time / total_dur,
                "pause_total_time": total_pause,
                "pause_ratio": total_pause / total_dur,
                "pause_count": pause_dur.size,
                "pause_frequency": pause_dur.size / total_dur * 60,
                "pause_duration_mean": safe_mean(pause_dur),
                "pause_duration_median": robust_median(pause_dur),
                "pause_duration_std": safe_std(pause_dur),
                "pause_longest": safe_max(pause_dur),
                "speech_burstiness": safe_mean(speech_dur),
            }
        )

        # слоговая огибающая
        bp_b, bp_a = sg.butter(2, [200, 4000], btype="bandpass", fs=sr)
        env = np.abs(hilbert(sg.filtfilt(bp_b, bp_a, y)))
        prom = 0.25 * median_abs_deviation(env) + EPS
        peaks, _ = find_peaks(env, prominence=prom, distance=int(0.08 * sr))

        syll_times = peaks * hop_len / sr
        syll_count = len(syll_times)
        if syll_count > 2:
            inter = np.diff(syll_times)
            npvi = 100 * np.mean(
                np.abs(np.diff(inter)) / ((inter[1:] + inter[:-1]) / 2 + EPS)
            )
            feats["nPVI_syllable"] = npvi
        else:
            feats["nPVI_syllable"] = 0.0

        speaking_rate = syll_count / total_dur * 60
        articulation_rate = syll_count / (speech_time + EPS) * 60
        feats.update(
            {
                "syllable_rate": speaking_rate,
                "speaking_rate": speaking_rate,
                "articulation_rate": articulation_rate,
            }
        )

        if syll_count > 1:
            intervals = np.diff(syll_times)
            cv = safe_std(intervals) / (safe_mean(intervals) + EPS)
            feats.update(
                {
                    "syllable_interval_cv": cv,
                    "rhythm_regularity": float(np.clip(1 - cv, 0.0, 1.0)),
                }
            )
        else:
            feats.update({"syllable_interval_cv": 0.0, "rhythm_regularity": 0.0})

        return feats

    # --------------- Голосовые признаки ---------------------------
    def _voice(
        self,
        y: np.ndarray,
        sr: int,
        frame_spec: FrameSpec,
        vad_mask: np.ndarray | None,
    ) -> Dict[str, float]:
        if crepe is None:
            # crepe недоступен — возвращаем нули, чтобы не рушить пайплайн
            zero_keys = [
                "F0_mean",
                "F0_median",
                "F0_std",
                "F0_range",
                "cv_F0",
                "jitter_local",
                "jitter_abs",
                "jitter_PPQ5",
                "shimmer_local",
                "shimmer_PPQ5",
                "HNR",
                "HNR_std",
                "NHR",
                "CPP_mean",
                "CPP_std",
                "H1_H2_mean",
                "breathiness_ratio",
                "creakiness_ratio",
                "voice_breaks_ratio",
                "voice_breaks_count",
                "shimmer_local_std",
                "jitter_local_std",
                "F0_mean_ST",
                "F0_std_ST",
                "F0_range_ST",
                "cv_F0_ST",
            ]
            return {k: 0.0 for k in zero_keys}

        _, yin_hop = frame_spec
        step_ms = yin_hop / sr * 1000
        _, f0_arr, conf, _ = crepe.predict(
            y.astype(np.float32),
            sr,
            model_capacity="tiny",
            step_size=step_ms,
            verbose=0,
        )
        f0 = f0_arr.flatten()
        voiced = conf.flatten() > 0.6

        if vad_mask is not None:
            min_len = min(len(voiced), len(vad_mask))
            voiced = voiced[:min_len] & vad_mask[:min_len]
            f0 = f0[:min_len]
            conf = conf[:min_len]

        idx_voiced = np.where(voiced)[0]
        if idx_voiced.size == 0:
            return {
                k: 0.0
                for k in [
                    "F0_mean",
                    "F0_median",
                    "F0_std",
                    "F0_range",
                    "cv_F0",
                    "jitter_local",
                    "jitter_abs",
                    "jitter_PPQ5",
                    "shimmer_local",
                    "shimmer_PPQ5",
                    "HNR",
                    "HNR_std",
                    "NHR",
                    "CPP_mean",
                    "CPP_std",
                    "H1_H2_mean",
                    "breathiness_ratio",
                    "creakiness_ratio",
                    "voice_breaks_ratio",
                    "voice_breaks_count",
                    "shimmer_local_std",
                    "jitter_local_std",
                    "F0_mean_ST",
                    "F0_std_ST",
                    "F0_range_ST",
                    "cv_F0_ST",
                ]
            }

        f0_voiced = f0[idx_voiced]
        st = 12 * np.log2(f0_voiced / 55.0)

        feats: Dict[str, float] = {}
        feats.update(
            {
                "F0_mean": safe_mean(f0_voiced),
                "F0_median": robust_median(f0_voiced),
                "F0_std": safe_std(f0_voiced),
                "F0_range": float(
                    np.percentile(f0_voiced, 90) - np.percentile(f0_voiced, 10)
                ),
                "F0_mean_ST": safe_mean(st),
                "F0_std_ST": safe_std(st),
                "F0_range_ST": float(np.percentile(st, 90) - np.percentile(st, 10)),
            }
        )
        feats["cv_F0"] = feats["F0_std"] / (feats["F0_mean"] + EPS)
        feats["cv_F0_ST"] = feats["F0_std_ST"] / (feats["F0_mean_ST"] + EPS)

        # jitter
        if len(idx_voiced) > 2:
            periods_sec = 1.0 / (f0_voiced + EPS)
            dP = np.abs(np.diff(periods_sec))
            feats["jitter_local"] = safe_mean(dP / (periods_sec[:-1] + EPS))
            feats["jitter_local_std"] = safe_std(dP / (periods_sec[:-1] + EPS))
            feats["jitter_abs"] = safe_mean(dP) * 1000
            if len(dP) > 4:
                T_roll = np.convolve(dP, np.ones(5) / 5, "valid")
                feats["jitter_PPQ5"] = safe_mean(
                    np.abs(dP[2:-2] - T_roll) / (T_roll + EPS)
                )

        # shimmer
        T_samp = np.diff(idx_voiced) * yin_hop
        amp = []
        for fr, T in zip(idx_voiced[:-1], T_samp):
            start = fr * yin_hop
            seg = y[start : start + int(T)]
            if len(seg) < 3:
                continue
            amp.append(np.sqrt(np.mean(seg**2)))
        amp = np.asarray(amp)
        if amp.size > 2:
            dA = np.abs(np.diff(amp))
            feats["shimmer_local"] = safe_mean(dA / (0.5 * (amp[1:] + amp[:-1]) + EPS))
            feats["shimmer_local_std"] = safe_std(
                dA / (0.5 * (amp[1:] + amp[:-1]) + EPS)
            )
            if amp.size > 6:
                kern = np.ones(5) / 5
                avg5 = np.convolve(amp, kern, "valid")
                feats["shimmer_PPQ5"] = safe_mean(
                    np.abs(amp[2:-2] - avg5) / (avg5 + EPS)
                )

        # HNR via parselmouth
        snd = parselmouth.Sound(y, sr)
        harr = snd.to_harmonicity_cc(
            time_step=yin_hop / sr, minimum_pitch=50, silence_threshold=0.1
        )
        hnr_vals = harr.values[harr.values > -200]
        feats["HNR"] = safe_mean(hnr_vals)
        feats["HNR_std"] = safe_std(hnr_vals)
        feats["NHR"] = (
            safe_mean(1 / (10 ** (hnr_vals / 10) + 1)) if hnr_vals.size else 0.0
        )

        # CPP
        cpp_vals = []
        win_len = 2 * yin_hop
        window = np.hanning(win_len)
        for fr_i in idx_voiced:
            start = fr_i * yin_hop
            raw = y[start : start + win_len]
            if raw.size < win_len:
                raw = np.pad(raw, (0, win_len - raw.size))
            seg = raw * window
            if win_len < 256:
                continue
            log_mag = np.log(np.abs(np.fft.rfft(seg)) + EPS)
            cep = np.fft.irfft(log_mag)
            que = np.arange(len(cep)) / sr
            mask = (que >= 0.002) & (que <= 0.02)
            if mask.any():
                cpp_vals.append(np.max(cep[mask]) - np.median(cep))
        cpp_vals = np.asarray(cpp_vals)
        feats["CPP_mean"] = safe_mean(cpp_vals)
        feats["CPP_std"] = safe_std(cpp_vals)

        # H1-H2
        h1h2 = []
        for fr_i in idx_voiced:
            cur_f0 = f0[fr_i]
            if np.isnan(cur_f0) or cur_f0 < 50:
                continue
            start = fr_i * yin_hop
            raw = y[start : start + win_len]
            if raw.size < win_len:
                raw = np.pad(raw, (0, win_len - raw.size))
            seg = raw * window
            spec = np.abs(np.fft.rfft(seg))
            freq = np.fft.rfftfreq(win_len, 1 / sr)
            h1_idx = np.argmin(np.abs(freq - cur_f0))
            h2_idx = np.argmin(np.abs(freq - 2 * cur_f0))
            h1h2.append(20 * np.log10((spec[h1_idx] + EPS) / (spec[h2_idx] + EPS)))
        feats["H1_H2_mean"] = safe_mean(np.asarray(h1h2))

        # breathiness / creakiness
        feats["breathiness_ratio"] = (
            float(np.mean(hnr_vals < 5)) if hnr_vals.size else 0.0
        )
        feats["creakiness_ratio"] = (
            float(np.mean(cpp_vals < 7)) if cpp_vals.size else 0.0
        )

        # voice breaks
        breaks, cur = [], 0
        for v in voiced.astype(int):
            if v == 0:
                cur += 1
            elif cur:
                breaks.append(cur)
                cur = 0
        if cur:
            breaks.append(cur)
        breaks_dur = np.asarray(breaks) * yin_hop / sr
        speech_time = len(idx_voiced) * yin_hop / sr + EPS
        feats["voice_breaks_count"] = len(breaks)
        feats["voice_breaks_ratio"] = breaks_dur.sum() / speech_time

        return feats

    # --------------- Формантные признаки --------------------------
    def _formants(
        self,
        y: np.ndarray,
        sr: int,
        vad_mask: np.ndarray | None,
    ) -> Dict[str, float]:
        zero_keys = [
            "F1_mean",
            "F2_mean",
            "F3_mean",
            "F1_std",
            "F2_std",
            "F3_std",
            "F1_median",
            "F2_median",
            "F3_median",
            "F2_to_F1_ratio_mean",
            "F3_to_F1_ratio_mean",
            "F3_to_F2_ratio_mean",
            "F2_F1_slope_mean",
            "F2_F1_slope_std",
            "formant_dispersion_mean",
            "formant_dispersion_std",
            "vowel_space_area_bark",
        ]
        feats: Dict[str, float] = {k: 0.0 for k in zero_keys}

        hop = int(sr * 0.01)
        win = int(sr * 0.03)
        snd = parselmouth.Sound(y, sr)
        formant_obj = snd.to_formant_burg(
            time_step=hop / sr,
            maximum_formant=5500,
            window_length=0.03,
            pre_emphasis_from=50,
        )

        n_frames = int(np.floor(len(y) / hop))
        times = (np.arange(n_frames) * hop + win / 2) / sr
        if vad_mask is not None:
            times = times[: len(vad_mask)][vad_mask[: len(times)]]
        if times.size == 0:
            return feats

        F1, F2, F3 = [], [], []
        for t in times:
            f1 = formant_obj.get_value_at_time(1, t)
            f2 = formant_obj.get_value_at_time(2, t)
            f3 = formant_obj.get_value_at_time(3, t)
            if 200 < f1 < 1000 and 500 < f2 < 3500 and np.isfinite([f1, f2, f3]).all():
                F1.append(f1)
                F2.append(f2)
                F3.append(f3)
        if not F1:
            return feats

        F1, F2, F3 = map(np.asarray, (F1, F2, F3))
        feats.update(
            {
                "F1_mean": safe_mean(F1),
                "F2_mean": safe_mean(F2),
                "F3_mean": safe_mean(F3),
                "F1_std": safe_std(F1),
                "F2_std": safe_std(F2),
                "F3_std": safe_std(F3),
                "F1_median": robust_median(F1),
                "F2_median": robust_median(F2),
                "F3_median": robust_median(F3),
            }
        )
        feats["F2_to_F1_ratio_mean"] = safe_mean(F2 / (F1 + EPS))
        feats["F3_to_F1_ratio_mean"] = safe_mean(F3 / (F1 + EPS))
        feats["F3_to_F2_ratio_mean"] = safe_mean(F3 / (F2 + EPS))

        if len(F1) > 1:
            slopes = np.diff(F2) / (np.diff(F1) + EPS)
            feats["F2_F1_slope_mean"] = safe_mean(slopes)
            feats["F2_F1_slope_std"] = safe_std(slopes)

        disp = np.sqrt((F3 - F2) ** 2 + (F2 - F1) ** 2)
        feats["formant_dispersion_mean"] = safe_mean(disp)
        feats["formant_dispersion_std"] = safe_std(disp)

        pts = np.column_stack([hz_to_bark(F1), hz_to_bark(F2)])
        feats["vowel_space_area_bark"] = (
            ConvexHull(pts).volume if len(pts) >= 3 else 0.0
        )

        return feats

    # --------------- Спектральные признаки ------------------------
    def _spectral(
        self,
        y: np.ndarray,
        sr: int,
        frame_spec: FrameSpec,
    ) -> Dict[str, float]:
        zero_keys = [
            "spectral_centroid",
            "spectral_centroid_std",
            "spectral_bandwidth",
            "spectral_bandwidth_std",
            "spectral_rolloff",
            "spectral_flatness",
            "spectral_rolloff_95",
            "spectral_entropy",
            "spectral_entropy_std",
            "spectral_slope",
            "spectral_flux",
            "spectral_delta",
            "alpha_ratio",
            "HF_ratio",
            "spectral_skewness",
            "spectral_kurtosis",
        ]
        for lo, hi in [(0, 1000), (1000, 3000), (3000, 6000)]:
            zero_keys.append(f"spectral_centroid_band{lo // 1000}_{hi // 1000}kHz")
        feats: Dict[str, float] = {k: 0.0 for k in zero_keys}

        fl, hl = frame_spec
        S = (
            np.abs(librosa.stft(y, n_fft=fl, hop_length=hl, window="hann", center=True))
            ** 2
        )
        freqs = librosa.fft_frequencies(sr=sr, n_fft=fl)

        frame_energy = np.sum(S, axis=0)
        good = frame_energy > 0
        S_good = S[:, good]

        centroid = librosa.feature.spectral_centroid(S=S_good, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(S=S_good, sr=sr)
        roll85 = librosa.feature.spectral_rolloff(S=S_good, sr=sr, roll_percent=0.85)
        flatness = librosa.feature.spectral_flatness(S=S_good)

        feats: Dict[str, float] = {}
        feats.update(
            {
                "spectral_centroid": safe_mean(centroid),
                "spectral_centroid_std": safe_std(centroid),
                "spectral_bandwidth": safe_mean(bandwidth),
                "spectral_bandwidth_std": safe_std(bandwidth),
                "spectral_rolloff": safe_mean(roll85),
                "spectral_flatness": safe_mean(flatness),
            }
        )
        roll95 = librosa.feature.spectral_rolloff(S=S_good, sr=sr, roll_percent=0.95)
        feats["spectral_rolloff_95"] = safe_mean(roll95)

        p = S_good / (np.sum(S_good, axis=0, keepdims=True) + EPS)
        ent = -np.sum(p * np.log2(p + EPS), axis=0) / np.log2(p.shape[0])
        feats["spectral_entropy"] = safe_mean(ent)
        feats["spectral_entropy_std"] = safe_std(ent)

        mag_db = 10 * np.log10(S_good + EPS)
        slopes = []
        for col in range(mag_db.shape[1]):
            m = mag_db[:, col]
            keep = m > (np.max(m) - 80)
            if np.sum(keep) < 3:
                continue
            x = freqs[keep] / 1000
            y_db = m[keep]
            diffs = np.subtract.outer(y_db, y_db)
            slopes_pair = diffs / (np.subtract.outer(x, x) + EPS)
            slopes.append(np.median(slopes_pair[np.isfinite(slopes_pair)]))
        feats["spectral_slope"] = safe_mean(np.asarray(slopes))

        S_norm = librosa.util.normalize(S_good, norm=1, axis=0)
        flux = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))
        feats["spectral_flux"] = safe_mean(flux)
        feats["spectral_delta"] = safe_mean(np.abs(np.diff(S_norm, axis=1)))

        def band_energy(lo: float, hi: float):
            idx = (freqs >= lo) & (freqs < hi)
            return np.sum(S_good[idx, :], axis=0)

        energy_low = band_energy(50, 1000)
        energy_mid = band_energy(1000, 5000)
        energy_hf = band_energy(4000, sr / 2)
        feats["alpha_ratio"] = safe_mean(energy_low / (energy_mid + EPS))
        feats["HF_ratio"] = safe_mean(energy_hf / (energy_low + EPS))

        bands = [(0, 1000), (1000, 3000), (3000, 6000)]
        for lo, hi in bands:
            idx = (freqs >= lo) & (freqs < hi)
            feats[f"spectral_centroid_band{lo // 1000}_{hi // 1000}kHz"] = safe_mean(
                np.sum(S_good[idx, :], axis=0)
            )

        flat = S_good.flatten()
        mu, sigma = np.mean(flat), np.std(flat) + EPS
        feats["spectral_skewness"] = float(np.mean(((flat - mu) / sigma) ** 3))
        feats["spectral_kurtosis"] = float(np.mean(((flat - mu) / sigma) ** 4) - 3)

        return feats

    # --------------- Артикуляционные признаки ---------------------
    def _articulation(
        self,
        y: np.ndarray,
        sr: int,
        frame_spec: FrameSpec,
    ) -> Dict[str, float]:
        zero_keys = [
            "vowel_reduction_index",
            "vowel_openness_index",
            "consonant_to_vowel_energy_ratio",
            "nasal_energy_ratio",
            "articulation_precision",
            "articulation_variability",
            "spectral_centroid_std",
            "spectral_centroid_skew",
            "spectral_centroid_kurt",
            "spectral_tilt_ts_mean",
            "spectral_tilt_ts_std",
            "sibilant_balance",
            "clarity_index",
            "speech_sharpness",
            "hard_to_soft_ratio",
            "palatalization_ratio",
            "yotization_strength",
        ]
        feats: Dict[str, float] = {k: 0.0 for k in zero_keys}

        fl, hl = frame_spec
        S = np.abs(
            librosa.stft(y, n_fft=fl, hop_length=hl, window="hann", center=False)
        )
        freqs = librosa.fft_frequencies(sr=sr, n_fft=fl)
        rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=hl, center=False)[0]
        sc = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        flat = librosa.feature.spectral_flatness(S=S)[0]
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)[0]

        feats: Dict[str, float] = {}
        energy_thr = np.percentile(rms, 20)
        vowels = (rms > energy_thr) & (sc < 2000) & (flat < 0.2)
        conson = ~vowels

        loud_v_sc = sc[vowels]
        quiet_sc = sc[~vowels]
        feats["vowel_reduction_index"] = (
            safe_mean(remove_outliers(quiet_sc))
            / (safe_mean(remove_outliers(loud_v_sc)) + EPS)
            if loud_v_sc.size
            else 0.0
        )
        feats["vowel_openness_index"] = robust_median(loud_v_sc) / 500

        energy_v = safe_mean(rms[vowels])
        energy_c = safe_mean(rms[conson])
        feats["consonant_to_vowel_energy_ratio"] = energy_c / (energy_v + EPS)

        low_idx = freqs < 300
        mid_idx = (freqs >= 300) & (freqs < 1200)
        nasal_ratio = safe_mean(
            np.sum(S[low_idx, :], 0) / (np.sum(S[mid_idx, :], 0) + EPS)
        )
        feats["nasal_energy_ratio"] = nasal_ratio

        feats["articulation_precision"] = 1 - safe_mean(flat)
        feats["articulation_variability"] = safe_std(flat)

        feats["spectral_centroid_std"] = safe_std(sc)
        feats["spectral_centroid_skew"] = (
            float(np.mean(((sc - np.mean(sc)) / (np.std(sc) + EPS)) ** 3))
            if sc.size
            else 0.0
        )
        feats["spectral_centroid_kurt"] = (
            float(np.mean(((sc - np.mean(sc)) / (np.std(sc) + EPS)) ** 4) - 3)
            if sc.size
            else 0.0
        )

        ts_slopes = []
        for col in range(S.shape[1]):
            mag_db = 20 * np.log10(S[:, col] + EPS)
            keep = mag_db > (np.max(mag_db) - 70)
            if keep.sum() < 5:
                continue
            slope, _, _, _ = theilslopes(mag_db[keep], freqs[keep] / 1000)
            ts_slopes.append(slope)
        ts_slopes = remove_outliers(np.asarray(ts_slopes))
        feats["spectral_tilt_ts_mean"] = safe_mean(ts_slopes)
        feats["spectral_tilt_ts_std"] = safe_std(ts_slopes)

        hi_band = (freqs >= 4000) & (freqs < 8000)
        mid_band = (freqs >= 1000) & (freqs < 4000)
        sib_ratio = np.sum(S[hi_band, :], 0) / (np.sum(S[mid_band, :], 0) + EPS)
        feats["sibilant_balance"] = safe_mean(sib_ratio)

        low_b = np.sum(S[freqs < 1000, :], 0) + EPS
        high_b = np.sum(S[freqs >= 2000, :], 0) + EPS
        clarity = np.log10(high_b / low_b)
        feats["clarity_index"] = safe_mean(clarity)
        feats["speech_sharpness"] = safe_mean(contrast)

        hard = np.sum(ts_slopes > -15)
        soft = np.sum(ts_slopes <= -15)
        feats["hard_to_soft_ratio"] = hard / (soft + EPS)

        sc_thr = np.percentile(sc, 75)
        pal_frames = np.sum(sc > sc_thr)
        pal_ratio = pal_frames / (len(sc) + EPS)
        pal_ratio_norm = pal_ratio / (np.max([pal_ratio, 0.15]) + EPS)
        clarity_pos = max(feats.get("clarity_index", 0.0), 0.0)
        feats["palatalization_ratio"] = pal_ratio
        feats["yotization_strength"] = float(
            np.clip(pal_ratio_norm * clarity_pos * 0.8, 0.0, 1.0)
        )

        return feats

    # --------------- Динамические признаки ------------------------
    def _dynamics(
        self,
        y: np.ndarray,
        sr: int,
        frame_spec: FrameSpec,
        vad_mask: np.ndarray | None,
    ) -> Dict[str, float]:
        zero_keys = [
            "energy_modulation_index",
            "intensity_range_db",
            "envelope_decay_rate",
            "amplitude_rhythm_regularity",
            "rhythm_stability_drift",
            "F0_slope_mean",
            "F0_slope_std",
            "F0_range_semitone",
            "cv_F0_dyn",
            "speech_burstiness_cv",
            "beat_strength_ratio",
        ]
        feats: Dict[str, float] = {k: 0.0 for k in zero_keys}

        fl, hl = frame_spec
        feats: Dict[str, float] = {}
        rms_frames = librosa.feature.rms(
            y=y, frame_length=fl, hop_length=hl, center=False
        )[0]
        feats["energy_modulation_index"] = safe_std(rms_frames) / (
            safe_mean(rms_frames) + EPS
        )
        feats["intensity_range_db"] = 20 * np.log10(
            np.percentile(rms_frames, 95) / (np.percentile(rms_frames, 5) + EPS)
        )

        t_rms = np.arange(len(rms_frames)) * hl / sr
        slope = (
            np.polyfit(t_rms, 20 * np.log10(rms_frames + EPS), 1)[0]
            if len(t_rms) > 1
            else 0.0
        )
        feats["envelope_decay_rate"] = slope

        env = np.abs(hilbert(y))
        b_lp, a_lp = sg.butter(2, 10, btype="low", fs=sr)
        env_lp = sg.filtfilt(b_lp, a_lp, env)
        thr = np.median(env_lp) + 0.25 * median_abs_deviation(env_lp)
        peaks, _ = find_peaks(env_lp, height=thr, distance=int(0.05 * sr))
        if peaks.size > 3:
            intervals = np.diff(peaks) / sr
            cv_int = safe_std(intervals) / (safe_mean(intervals) + EPS)
            feats["amplitude_rhythm_regularity"] = 1 - cv_int
            half = len(intervals) // 2
            early = intervals[:half]
            late = intervals[half:]
            early_cv = safe_std(early) / (safe_mean(early) + EPS)
            late_cv = safe_std(late) / (safe_mean(late) + EPS)
            feats["rhythm_stability_drift"] = late_cv - early_cv
        else:
            feats["amplitude_rhythm_regularity"] = 0.0
            feats["rhythm_stability_drift"] = 0.0

        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=hl)
        voiced = ~np.isnan(f0)
        if vad_mask is not None:
            m = min(len(voiced), len(vad_mask))
            voiced = voiced[:m] & vad_mask[:m]
            f0 = f0[:m]
        f0_v = f0[voiced]
        if f0_v.size:
            st = 12 * np.log2(f0_v / 55.0)
            dt = np.diff(st)
            feats["F0_slope_mean"] = safe_mean(dt) * sr / hl
            feats["F0_slope_std"] = safe_std(dt) * sr / hl
            feats["F0_range_semitone"] = float(
                np.percentile(st, 90) - np.percentile(st, 10)
            )
            feats["cv_F0_dyn"] = safe_std(f0_v) / (safe_mean(f0_v) + EPS)
        else:
            feats.update(
                {
                    "F0_slope_mean": 0.0,
                    "F0_slope_std": 0.0,
                    "F0_range_semitone": 0.0,
                    "cv_F0_dyn": 0.0,
                }
            )

        speech_mask = (
            vad_mask
            if vad_mask is not None
            else rms_frames > np.percentile(rms_frames, 20)
        )
        speech_blocks = []
        cur = 0
        for v in speech_mask:
            if v:
                cur += 1
            elif cur:
                speech_blocks.append(cur * hl / sr)
                cur = 0
        if cur:
            speech_blocks.append(cur * hl / sr)
        bursts = np.asarray(speech_blocks)
        feats["speech_burstiness_cv"] = (
            safe_std(bursts) / (safe_mean(bursts) + EPS) if bursts.size else 0.0
        )

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        feats["beat_strength_ratio"] = safe_mean(onset_env) / (
            safe_max(onset_env) + EPS
        )

        return feats


# ---------------------------------------------------------------------------
# Унифицированный класс для окнами, интегрированный с BaseExtractor
# ---------------------------------------------------------------------------
class AcousticExtractor(BaseExtractor):
    """
    Унифицированный интерфейс для акустического анализа речи с режимами:
      • analysis_mode='non_cumulative' — окна [t, t+step_size]
      • analysis_mode='cumulative'     — окна [0, t]
      • analysis_mode='full'           — один кадр на весь файл [0, T]

    Возвращает DataFrame с колонками: start_sec, end_sec, признаки
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size: float = 5.0,
        step_size: float = 2.0,
        analysis_mode: Literal[
            "non_cumulative", "cumulative", "full"
        ] = "non_cumulative",
        use_vad: bool = True,
    ):
        super().__init__(
            sample_rate=sample_rate, window_size=window_size, step_size=step_size
        )
        # Инициализация вашего исходного вычислителя признаков
        self._fe = FeatureExtractor(use_vad=use_vad)  # type: ignore[name-defined]
        self.analysis_mode = analysis_mode

    # ------------------------------------------------------------------
    # Основные методы: массив → DataFrame, путь → DataFrame, датасет → CSV
    # ------------------------------------------------------------------
    def extract_from_array(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """
        Анализирует по входному массиву и частоте дискретизации.
        """
        if self.analysis_mode == "full":
            df = self._fe.extract_from_array(y, sr)
            df["start_sec"] = 0.0
            df["end_sec"] = len(y) / sr
            return df

        frames: List[pd.DataFrame] = []
        if self.analysis_mode == "non_cumulative":
            # прежняя логика: скользящие/непересекающиеся окна
            for start, end in self.iter_windows(y, sr):
                segment = y[start:end]
                df = self._fe.extract_from_array(segment, sr)
                df["start_sec"] = start / sr
                df["end_sec"] = end / sr
                frames.append(df)
        elif self.analysis_mode == "cumulative":
            # накопительные окна: [0, t]
            total = len(y)
            win = int(self.window_size * sr)
            step = int(self.step_size * sr)
            if win <= 0:
                win = int(1.0 * sr)
            if step <= 0:
                step = win
            t = win
            while t <= total:
                segment = y[:t]
                df = self._fe.extract_from_array(segment, sr)
                df["start_sec"] = 0.0
                df["end_sec"] = t / sr
                frames.append(df)
                t += step
        else:
            raise ValueError(f"Unknown analysis_mode: {self.analysis_mode}")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def extract_from_path(self, path: str | Path) -> pd.DataFrame:
        """
        Анализирует аудио по входному пути до файла.
        """
        y, sr = self.load_audio(path)
        return self.extract_from_array(y, sr)

    def extract_dataset(
        self,
        input_dir: str | Path,
        output_csv: str | Path,
        verbose: bool = True,
    ) -> None:
        """
        Обрабатывает директорию файлов по заданному режиму при инициализации AcousticExtractor.
        Параметры window_size/step_size и analysis_mode применяются к каждому файлу.
        Принимает на вход путь до директории с анализируемыми файлами и путь, куда сложить результат (формат .csv указать вручную).
        """
        from tqdm.auto import tqdm

        input_dir, output_csv = Path(input_dir), Path(output_csv)
        files = sorted(
            [
                p
                for p in Path(input_dir).glob("*")
                if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a")
            ]
        )
        all_rows: List[pd.DataFrame] = []
        iterator = tqdm(files, desc="Acoustic") if verbose else files

        for p in iterator:
            try:
                df = self.extract_from_path(p)
                if not df.empty:
                    df.insert(0, "file", p.name)
                    all_rows.append(df)
            except Exception as e:
                print(f"⚠️ Ошибка при обработке {p.name}: {e}")

        if not all_rows:
            raise RuntimeError(
                "Не удалось извлечь акустические признаки ни из одного файла."
            )

        result = pd.concat(all_rows, ignore_index=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        if verbose:
            print(f"✅ Сохранено {len(result)} окон в {output_csv}")
