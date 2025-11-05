import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Tuple

import librosa
import numpy as np
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="IProgress not found", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated as an API", category=UserWarning
)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("stanza").setLevel(logging.ERROR)
logging.getLogger("stanza.resources").setLevel(logging.ERROR)


class BaseExtractor(ABC):
    """
    Базовый класс для всех экстракторов.
    Обеспечивает единый интерфейс загрузки и предобработки аудио.
    """

    def __init__(
        self, sample_rate: int = 16000, window_size: float = 5.0, step_size: float = 2.0
    ):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.step_size = step_size

    # ----------------------- базовая загрузка -----------------------
    def load_audio(self, source: str | Path) -> Tuple[np.ndarray, int]:
        """Загрузка и нормализация аудио любого формата."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(str(source), sr=self.sample_rate, mono=True)
        y = librosa.util.normalize(y)
        return y, sr

    # ------------------------ работа с окнами ------------------------
    def iter_windows(self, y: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """Генератор окон [start, end] в сэмплах."""
        win_len = int(self.window_size * sr)
        step_len = int(self.step_size * sr)
        total_len = len(y)
        windows = []
        for start in range(0, total_len, step_len):
            end = min(start + win_len, total_len)
            if (end - start) >= 0.5 * win_len:
                windows.append((start, end))
        return windows

    # ------------------------ абстрактные методы ------------------------
    @abstractmethod
    def extract_from_array(self, y: np.ndarray, sr: int) -> Any:
        """Извлекает признаки из аудиомассива."""
        pass

    def extract_from_path(self, path: str | Path) -> Any:
        """Загружает аудио и вызывает извлечение признаков."""
        y, sr = self.load_audio(path)
        return self.extract_from_array(y, sr)
