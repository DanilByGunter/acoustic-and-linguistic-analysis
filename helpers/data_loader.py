import json
from pathlib import Path
from typing import Dict, List, Union

from catboost import CatBoostClassifier


class ResourceLoader:
    """
    Централизованная загрузка словарей, моделей и вспомогательных данных.
    Работает с директориями ресурсов, не жёстко привязанными к путям.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def load_json(
        self, name: str
    ) -> Union[
        Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, List[str]]], List[str]
    ]:
        """
        Загружает JSON-словарь по имени без расширения.
        Пример: load_json("modal_verb") -> base_dir/modal_verb.json
        """
        path = self.base_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"JSON '{name}' не найден в {self.base_dir}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_catboost(self, name: str) -> CatBoostClassifier:
        """
        Загружает CatBoostClassifier по имени без расширения.
        Пример: load_catboost("catboost_modalities") -> base_dir/catboost_modalities.cbm
        """
        path = self.base_dir / f"{name}.cbm"
        if not path.exists():
            raise FileNotFoundError(
                f"CatBoost модель '{name}' не найдена в {self.base_dir}"
            )
        model = CatBoostClassifier()
        model.load_model(str(path))
        return model

    def check_exists(self, name: str) -> bool:
        """
        Проверяет наличие ресурса (json или cbm) по имени.
        """
        json_path = self.base_dir / f"{name}.json"
        cbm_path = self.base_dir / f"{name}.cbm"
        return json_path.exists() or cbm_path.exists()

    def list_resources(self) -> List[str]:
        """Возвращает список доступных ресурсов (без расширений)."""
        files = list(self.base_dir.glob("*.json")) + list(self.base_dir.glob("*.cbm"))
        return [f.stem for f in files]
