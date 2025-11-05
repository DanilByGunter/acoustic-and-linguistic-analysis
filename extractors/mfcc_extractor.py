from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Timeline
from torchaudio.functional import compute_deltas
from torchaudio.transforms import MelSpectrogram
from tqdm.auto import tqdm
from transformers.utils import logging as hf_logging

from helpers.base_extractor import BaseExtractor

warnings.filterwarnings(
    "ignore",
    message="Lightning automatically upgraded your loaded checkpoint",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message="Model was trained with pyannote.audio", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="Model was trained with torch", category=UserWarning
)
hf_logging.set_verbosity_error()
logging.getLogger().setLevel(logging.ERROR)


# =============================================================================
#                                FeatureExtractor
# =============================================================================
class FeatureExtractor:
    """
    Находит MFCC + Δ + Δ².
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 64,
        device: Optional[torch.device] = None,
    ):
        self.sample_rate = sample_rate
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
            f_min=20,
            f_max=sample_rate // 2,
            power=2.0,
        ).to(self.device)

    @torch.inference_mode()
    def compute_mfcc_triplet(
        self, wav: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Принимает тензор формы [1, T] или [T], возвращает (mel_log, delta, delta2).
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel(wav)  # [1, n_mels, frames]
        mel = torch.log(mel + 1e-6)
        d1 = compute_deltas(mel)
        d2 = compute_deltas(d1)
        return mel, d1, d2

    @torch.inference_mode()
    def extract_from_array(self, wav_np: np.ndarray, sr: int) -> Dict[str, float]:
        """
        MFCC(+Δ,+Δ²) по всему массиву (усреднение по времени).
        """
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                torch.tensor(wav_np, dtype=torch.float32), sr, self.sample_rate
            )
        else:
            wav = torch.tensor(wav_np, dtype=torch.float32)
        wav = wav.to(self.device)
        mel, d1, d2 = self.compute_mfcc_triplet(wav)
        mel_mean = mel.mean(dim=-1).squeeze().detach().cpu().numpy()
        d1_mean = d1.mean(dim=-1).squeeze().detach().cpu().numpy()
        d2_mean = d2.mean(dim=-1).squeeze().detach().cpu().numpy()

        feats: Dict[str, float] = {}
        for i, v in enumerate(mel_mean, start=1):
            feats[f"MFCC_{i}"] = float(v)
        for i, v in enumerate(d1_mean, start=1):
            feats[f"Delta1_{i}"] = float(v)
        for i, v in enumerate(d2_mean, start=1):
            feats[f"Delta2_{i}"] = float(v)
        return feats


# =============================================================================
#                                 MFCCExtractor
# =============================================================================
class MFCCExtractor(BaseExtractor):
    """
    Унифицированный интерфейс для извлечения MFCC(+Δ,+Δ²) с диаризацией (PyAnnote) и VAD:
      • analysis_mode='non_cumulative' — окна [t, t+step_size]
      • analysis_mode='cumulative'     — окна [0, t]
      • analysis_mode='full'           — один кадр на весь файл [0, T]

    Возвращает DataFrame с колонками: start_sec, end_sec, MFCC_*, Delta1_*, Delta2_*.
    """

    def __init__(
        self,
        hf_token: str,
        sample_rate: int = 16_000,
        n_mels: int = 64,
        window_size: float = 5.0,
        step_size: float = 2.0,
        analysis_mode: Literal[
            "non_cumulative", "cumulative", "full"
        ] = "non_cumulative",
        device: Optional[torch.device] = None,
    ):
        super().__init__(sample_rate, window_size, step_size)
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Диаризация и VAD

        self.diar_pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        ).to(self.device)
        self.vad_pipe = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection", use_auth_token=hf_token
        ).to(self.device)

        # Внутренний вычислитель MFCC(+Δ,+Δ²)
        self._fe = FeatureExtractor(
            sample_rate=sample_rate, n_mels=n_mels, device=self.device
        )

        self.analysis_mode = analysis_mode

    # ------------------------------------------------------------------
    # Вспомогательные I/O
    # ------------------------------------------------------------------
    def _load_audio(self, path: Path) -> Tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0), sr

    def _resample_if_needed(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            return resampler(wav)
        return wav

    def _earliest_speaker(self, diarization) -> Optional[str]:
        first_seg, first_label = None, None
        for segment, _, label in diarization.itertracks(yield_label=True):
            if first_seg is None or segment.start < first_seg.start:
                first_seg, first_label = segment, label
        return first_label

    def _intersect_timelines(self, t1: Timeline, t2: Timeline) -> Timeline:
        return t1.crop(t2, mode="intersection")

    # ------------------------------------------------------------------
    # Агрегация речевого аудио по таймлайну
    # ------------------------------------------------------------------
    def _collect_speech_segments(
        self, wav_path: Path
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Возвращает массив аудио (полный, в self.sample_rate) и список речевых интервалов [(start, end), ...]
        для выбранного (первого) говорящего, пересечённый с VAD.
        """
        diar = self.diar_pipe(wav_path)
        spk = self._earliest_speaker(diar)
        if spk is None:
            return np.zeros(0, dtype=np.float32), []

        tl_spk = diar.label_timeline(spk)
        vad = self.vad_pipe(wav_path)
        tl_speech = vad.get_timeline()
        final_tl = self._intersect_timelines(tl_spk, tl_speech)
        if not final_tl:
            return np.zeros(0, dtype=np.float32), []

        wav, orig_sr = self._load_audio(wav_path)
        wav = self._resample_if_needed(wav, orig_sr)
        wav_np = wav.cpu().numpy()

        speech_intervals: List[Tuple[float, float]] = []
        for seg in final_tl:
            speech_intervals.append((float(seg.start), float(seg.end)))
        return wav_np, speech_intervals

    # ------------------------------------------------------------------
    # Основные режимы анализа
    # ------------------------------------------------------------------
    def _analyze_full(
        self, wav_np: np.ndarray, sr: int, speech_intervals: List[Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Один кадр [0, T]: аккумулируем только речевые участки и считаем MFCC(+Δ,+Δ²).
        """
        if not speech_intervals:
            return pd.DataFrame()

        chunks = []
        for st, en in speech_intervals:
            s = int(st * sr)
            e = int(en * sr)
            if e > s:
                chunks.append(wav_np[s:e])
        if not chunks:
            return pd.DataFrame()

        concat = np.concatenate(chunks)
        feats = self._fe.extract_from_array(concat, sr)
        row = {"start_sec": 0.0, "end_sec": float(len(wav_np) / sr)}
        row.update(feats)
        return pd.DataFrame([row])

    def _analyze_cumulative(
        self, wav_np: np.ndarray, sr: int, speech_intervals: List[Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Накопительные окна [0, t] по речевым участкам (до t).
        """
        if not speech_intervals:
            return pd.DataFrame()

        duration = len(wav_np) / sr
        rows: List[Dict[str, float]] = []
        win = self.window_size
        step = self.step_size
        t = win
        while t <= max(win, duration) + 1e-9:
            # собираем аудио только из интервалов, чьи концы <= t
            parts = []
            for st, en in speech_intervals:
                if st >= t:
                    break
                seg_end = min(en, t)
                if seg_end > st:
                    s = int(st * sr)
                    e = int(seg_end * sr)
                    if e > s:
                        parts.append(wav_np[s:e])
            if parts:
                concat = np.concatenate(parts)
                feats = self._fe.extract_from_array(concat, sr)
                row = {"start_sec": 0.0, "end_sec": float(t)}
                row.update(feats)
                rows.append(row)
            t += step
            if t > duration and t - step >= duration:
                break

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _analyze_non_cumulative(
        self, wav_np: np.ndarray, sr: int, speech_intervals: List[Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Скользящие окна [t, t+win] внутри каждого речевого интервала.
        """
        if not speech_intervals:
            return pd.DataFrame()

        rows: List[Dict[str, float]] = []
        for st, en in speech_intervals:
            start = int(st * sr)
            end = int(en * sr)
            segment = wav_np[start:end]
            if len(segment) < sr * 0.3:
                continue

            for w_start, w_end in self.iter_windows(segment, sr):
                frame = segment[w_start:w_end]
                feats = self._fe.extract_from_array(frame, sr)
                row = {"start_sec": st + w_start / sr, "end_sec": st + w_end / sr}
                row.update(feats)
                rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Публичные методы BaseExtractor
    # ------------------------------------------------------------------
    def extract_from_path(self, path: str | Path) -> pd.DataFrame:
        """
        Анализирует аудио по входному пути до файла.
        """
        path = Path(path)
        wav_np, speech_intervals = self._collect_speech_segments(path)
        if wav_np.size == 0:
            return pd.DataFrame()

        if self.analysis_mode == "full":
            return self._analyze_full(wav_np, self.sample_rate, speech_intervals)
        elif self.analysis_mode == "cumulative":
            return self._analyze_cumulative(wav_np, self.sample_rate, speech_intervals)
        elif self.analysis_mode == "non_cumulative":
            return self._analyze_non_cumulative(
                wav_np, self.sample_rate, speech_intervals
            )
        else:
            raise ValueError(f"Unknown analysis_mode: {self.analysis_mode}")

    def extract_from_array(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """
        Анализирует по входному массиву и частоте дискретизации.
        """
        if self.analysis_mode == "full":
            feats = self._fe.extract_from_array(y, sr)
            row = {"start_sec": 0.0, "end_sec": len(y) / sr}
            row.update(feats)
            return pd.DataFrame([row])
        elif self.analysis_mode == "cumulative":
            rows: List[Dict[str, float]] = []
            total = len(y)
            win = int(self.window_size * sr)
            step = int(self.step_size * sr)
            if win <= 0:
                win = int(1.0 * sr)
            if step <= 0:
                step = win
            t = win
            while t <= total:
                seg = y[:t]
                feats = self._fe.extract_from_array(seg, sr)
                rows.append({"start_sec": 0.0, "end_sec": t / sr, **feats})
                t += step
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        else:  # non_cumulative
            rows: List[Dict[str, float]] = []
            for start, end in self.iter_windows(y, sr):
                seg = y[start:end]
                feats = self._fe.extract_from_array(seg, sr)
                rows.append({"start_sec": start / sr, "end_sec": end / sr, **feats})
            return pd.DataFrame(rows) if rows else pd.DataFrame()

    def extract_dataset(
        self,
        input_dir: str | Path,
        output_csv: str | Path,
        verbose: bool = True,
    ) -> None:
        """
        Обрабатывает директорию файлов по заданному режиму при инициализации MFCCExtractor.
        Параметры window_size/step_size и analysis_mode применяются к каждому файлу.
        Принимает на вход путь до директории с анализируемыми файлами и путь, куда сложить результат (формат .csv указать вручную).
        """
        input_dir, output_csv = Path(input_dir), Path(output_csv)
        files = sorted(
            [
                p
                for p in Path(input_dir).glob("*")
                if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a")
            ]
        )
        all_rows: List[pd.DataFrame] = []
        iterator = tqdm(files, desc="MFCC") if verbose else files

        for p in iterator:
            try:
                df = self.extract_from_path(p)
                if not df.empty:
                    df.insert(0, "file", p.name)
                    all_rows.append(df)
            except Exception as e:
                print(f"⚠️ Ошибка при обработке {p.name}: {e}")

        if not all_rows:
            raise RuntimeError("Не удалось извлечь признаки ни из одного файла.")

        result = pd.concat(all_rows, ignore_index=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        if verbose:
            print(f"✅ Сохранено {len(result)} окон речи в {output_csv}")
