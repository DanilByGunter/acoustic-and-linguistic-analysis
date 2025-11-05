from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import pymorphy3
import soundfile as sf
import stanza
import torch
from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Segmenter,
)
from pyaspeller import YandexSpeller
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModel,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
from transformers.utils import logging as hf_logging

from helpers.base_extractor import BaseExtractor
from helpers.data_loader import ResourceLoader

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="IProgress not found", category=UserWarning)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("stanza").setLevel(logging.ERROR)
logging.getLogger("stanza.resources").setLevel(logging.ERROR)


EPS = 1e-9


# ===========================================================================
#                              FeatureExtractor
# ===========================================================================
class FeatureExtractor:
    """
    Извлекает лингвистические признаки из аудио или текста.
    Содержит встроенные методы export_* для анализа речи.
    """

    def __init__(self, resources: ResourceLoader, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.words_per_sec = 2.0  # эвристика для текста: 2 слова ~ 1 сек

        # --- загрузка моделей ---
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            "antony66/whisper-large-v3-russian",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(
            "antony66/whisper-large-v3-russian"
        )
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.whisper,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            dtype=torch.bfloat16,
            device=self.device,
            ignore_warning=True,
        )

        self.morph = pymorphy3.MorphAnalyzer()
        self.speller = YandexSpeller()
        self.stanza_nlp = stanza.Pipeline(
            lang="ru", processors="tokenize,pos,lemma,depparse"
        )

        self.rubert_tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rubert-tiny2"
        )
        self.rubert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2").to(
            self.device
        )

        self.pipe_emotions = pipeline(
            "text-classification",
            model="MaxKazak/ruBert-base-russian-emotion-detection",
            device=0 if self.device == "cuda" else -1,
        )

        # --- словари / модели ---
        self.catboost_modalities = resources.load_catboost("catboost_modalities")
        self.banned_words = set(resources.load_json("banned_words_set"))
        self.freq_words = resources.load_json("frequency_words")
        self.modal_verbs = resources.load_json("modal_verb")
        self.all_modal_verbs = resources.load_json("all_modal_verbs")
        self.verbs_clusters = resources.load_json("verbs_clusters")
        self.negative_particles = resources.load_json("negative_particles")
        self.semantic_clusters = resources.load_json("semantic_clusters")
        self.conjunctions = resources.load_json("conjunctions")
        self.conjunctions_set = resources.load_json("conjunctions_set")
        self.coordinating_conjunctions = self.conjunctions.get(
            "COORDINATING_CONJUNCTIONS"
        )
        self.subodinating_conjunctions = self.conjunctions.get(
            "SUBORDINATING_CONJUNCTIONS"
        )
        self.other_linkers = self.conjunctions.get("OTHER_LINKERS")

        # Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)

    # ===========================================================================
    #                               Вспомогательные
    # ===========================================================================
    def _load_audio_mono(
        self, path: Path, target_sr: int = 16000
    ) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        if y.size:
            y = (y / (np.max(np.abs(y)) + 1e-12)).astype(np.float32)
        else:
            y = np.zeros(1, dtype=np.float32)
        return y, sr

    def _process_long_audio_vad(
        self,
        file_path: Path,
        max_chunk_length_ms: int = 90000,
        min_silence_len: int = 500,
        silence_thresh: int = -40,
    ) -> List[Dict[str, Any]]:
        """
        Разбивает длинное аудио по голосовой активности (VAD)
        и прогоняет через Whisper ASR с временными метками.
        Возвращает список сегментов [{'text': str, 'timestamp': (start, end)}].
        """
        # Загрузка
        audio = AudioSegment.from_file(file_path, format="wav")
        duration_ms = len(audio)

        # Детектируем речевые участки
        speech_segments = detect_nonsilent(
            audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
        )

        # Объединяем короткие участки в чанки не длиннее max_chunk_length_ms
        chunks = []
        current_start, current_end = 0, 0
        for start, end in speech_segments:
            if (start - current_end > min_silence_len * 2) or (
                (end - current_start) > max_chunk_length_ms
            ):
                if current_end > current_start:
                    chunks.append((current_start, current_end))
                current_start = start
            current_end = end
        if current_end > current_start:
            chunks.append((current_start, current_end))

        results = []
        for i, (start, end) in enumerate(chunks):
            chunk = audio[start:end]
            if len(chunk) < 800:  # менее 0.8 сек — пропускаем
                continue

            temp_name = f"temp_chunk_{i}.wav"
            chunk.export(temp_name, format="wav")

            try:
                result = self.asr_pipeline(
                    temp_name,
                    generate_kwargs={"language": "russian", "max_new_tokens": 256},
                    return_timestamps=True,
                )
                if "text" in result:
                    results.append(
                        {
                            "text": result["text"],
                            "timestamp": (start / 1000.0, end / 1000.0),
                        }
                    )
            except Exception as e:
                print(f"⚠️ Ошибка обработки чанка {i}: {e}")
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)

        # Если ничего не найдено — fallback
        if not results:
            return [{"text": "", "timestamp": (0.0, duration_ms / 1000.0)}]

        return results

    def _transcribe_full(self, audio_path: Path) -> Dict[str, Any]:
        """Полная транскрипция файла (единый вызов ASR). Возвращает {'text': str, 'chunks': [...]}."""
        segments = self._process_long_audio_vad(audio_path)
        full_text = " ".join([s["text"] for s in segments])
        return {"text": full_text, "chunks": segments}

    def _transcribe_segment(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Транскрипция отдельного сегмента аудио."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(tmp_path, y, sr)
        try:
            result = self._process_long_audio_vad(tmp_path)
            text = " ".join([r["text"] for r in result])
            return {"text": text, "chunks": result}
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    def _normalize_text(self, text: str) -> str:
        """Исправляет орфографию и очищает текст."""
        try:
            corrected = self.speller.spelled(text)
            corrected = re.sub(r"[^а-яА-ЯёЁ\w\s.,!?-]", "", corrected)
        except Exception as e:
            print("⚠️ Error processing with speller test.wav:", e)
            corrected = re.sub(r"[^а-яА-ЯёЁ\w\s.,!?-]", "", corrected)
        return corrected.strip()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[\w|-]+", text.lower(), re.UNICODE)

    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Возвращает эмбеддинги для предложений"""

        encoded_input = self.rubert_tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.rubert_model(**encoded_input)

        token_embeddings = model_output[0]
        input_mask_expanded = (
            encoded_input["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings.cpu()

    def _get_word_embedding(self, word: str, context: str) -> np.ndarray:
        """Возвращает контекстуальный эмбеддинг"""
        start_char = context.find(word)
        encoded_input = self.rubert_tokenizer(
            context,
            return_tensors="pt",
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mappings = encoded_input.pop("offset_mapping")[0].tolist()

        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            hidden_states = self.rubert_model(
                **encoded_input
            ).last_hidden_state.squeeze(0)

        if start_char == -1:
            if len(hidden_states) == 0:
                return np.zeros((hidden_states.size(-1),), dtype=np.float32)
            return hidden_states.mean(dim=0).numpy()

        end_char = start_char + len(word)
        valid_indices = []
        for i, (st_sub, end_sub) in enumerate(offset_mappings):
            if (
                (st_sub, end_sub) != (0, 0)
                and end_sub > start_char
                and st_sub < end_char
            ):
                valid_indices.append(i)

        if valid_indices:
            embedding = hidden_states[valid_indices].mean(dim=0)
        else:
            embedding = hidden_states.mean(dim=0)
        return embedding.cpu()

    def _find_conjunction_types(
        self,
        words: List[str],
        relation: Dict[str, List[str]],
        return_set: bool = True,
        return_dict: bool = False,
        use_morph: bool = False,
    ) -> List[str] | Dict[str, List[str]]:
        """
        Для поиска н-грамм, в том числе и сложных сочинительных или подчинительных союзов
        """
        if use_morph:
            words = [self.morph.parse(word)[0].normal_form for word in words]

        used_parts = {}
        found_types_set, found_types_list, found_types_dict = (
            set(),
            list(),
            defaultdict(list),
        )

        def check_compound_conjunction(conj, words_local):
            if "…" in conj:
                parts = conj.split("…")
                parts = [part.strip().split() for part in parts]
                for i in range(len(words_local)):
                    if i + len(parts[0]) <= len(words_local):
                        first_part_match = all(
                            words_local[i + j].lower() == parts[0][j].lower()
                            for j in range(len(parts[0]))
                        )
                        if first_part_match:
                            for k in range(i + len(parts[0]), len(words_local)):
                                if k + len(parts[1]) <= len(words_local):
                                    second_part_match = all(
                                        words_local[k + j].lower()
                                        == parts[1][j].lower()
                                        for j in range(len(parts[1]))
                                    )
                                    if second_part_match:
                                        part_key = tuple(parts[0] + parts[1])
                                        used_parts[part_key] = True
                                        return True
            else:
                conj_parts = conj.split()
                for i in range(len(words_local) - len(conj_parts) + 1):
                    if all(
                        words_local[i + j].lower() == conj_parts[j].lower()
                        for j in range(len(conj_parts))
                    ):
                        for used_part in used_parts:
                            if any(conj.lower() == part.lower() for part in used_part):
                                return False
                        return True
            return False

        for conj_type, conjunctions_list in relation.items():
            found_types_dict[conj_type] = []
            conjunctions_list = sorted(
                conjunctions_list,
                key=lambda x: (x.count(" ") + 1) * len(x),
                reverse=True,
            )
            for c in conjunctions_list:
                if check_compound_conjunction(c, words):
                    found_types_set.add(conj_type)
                    found_types_list += c.split(" ")
                    found_types_dict[conj_type] += c.split(" ")
                    break

        if return_set:
            return list(found_types_set)
        elif return_dict:
            return dict(found_types_dict)
        else:
            return found_types_list

    def _get_token_important(self, tokens: List[str]) -> Tuple[List[str], float]:
        tokens_important = [
            token
            for token in tokens
            if self.morph.parse(token)[0].tag.POS
            in [
                "NOUN",
                "VERB",
                "INFN",
                "ADJF",
                "ADJS",
                "ADVB",
                "NUMR",
                "NPRO",
                "PRTF",
                "PRTS",
                "GRND",
            ]
        ]
        cnt_tokens_important = len(tokens_important) or 1
        return tokens_important, cnt_tokens_important

    # ===========================================================================
    #                            Методы извлечения признаков
    # ===========================================================================
    def export_mean_sentence_length(
        self, sentences_tokens: List[List[str]]
    ) -> Tuple[int, float, float]:
        num_sentences = len(sentences_tokens)
        if num_sentences == 0:
            return 0, 0.0, 0.0

        token_counts = [len(s) for s in sentences_tokens]
        avg_sentence_length = sum(token_counts) / num_sentences
        ratio = avg_sentence_length / (10.38 * 2)
        return num_sentences, avg_sentence_length, min(ratio, 1.0)

    def export_mean_word_len(self, tokens: List[str]) -> Tuple[float, float]:
        if len(tokens) == 0:
            return 0.0, 0.0
        lengths = [len(t) for t in tokens]
        avg_token_length = sum(lengths) / len(lengths)
        ratio = avg_token_length / (5.28 * 2)
        return avg_token_length, min(ratio, 1.0)

    def export_count_unintelligible_words(
        self, primary_tokens: List[str], corrected_tokens: List[str]
    ) -> Tuple[int, int, float]:
        len_p, len_c = len(primary_tokens), len(corrected_tokens)
        if len_p == 0 and len_c == 0:
            return 0, 0, 0.0
        if len_p == len_c:
            # Совпадающие
            errors = sum(1 for p, c in zip(primary_tokens, corrected_tokens) if p == c)
            return len_p, len_c, errors / len_p if len_p else 0.0
        else:
            # При разной длине
            ratio = (min(len_p, len_c) / max(len_p, len_c)) ** 2
            return len_p, len_c, ratio

    def export_speed_speech(self, segments: List[Dict]) -> Tuple[float, float]:
        words_count = 0
        timestamps = []
        for seg in segments:
            seconds1, mseconds1 = map(float, f"{seg['timestamp'][0]:.2f}".split("."))
            seconds2, mseconds2 = map(float, f"{seg['timestamp'][1]:.2f}".split("."))
            time1 = seconds1 * 60 + round(mseconds1, 2)
            time2 = seconds2 * 60 + round(mseconds2, 2)
            timestamps.append(time2 - time1)
            words_count += len(re.findall(r"[\w|-]+", seg["text"].strip(), re.UNICODE))

        total_time_hours = sum(timestamps) / 3600 if sum(timestamps) > 0 else 1e-9
        speed_speech_rate = words_count / total_time_hours
        ratio = speed_speech_rate / (120 * 2)
        return speed_speech_rate, min(ratio, 1.0)

    def export_type_token_ratio(self, tokens: List[str]) -> Tuple[int, float]:
        if not tokens:
            return 0, 0.0
        unique_count = len(set(tokens))
        ttr = unique_count / len(tokens)
        return unique_count, ttr

    def export_classify_pronouns(
        self, tokens: List[str], cnt_tokens_important: int
    ) -> Tuple[Dict[str, int], Dict[str, float], float]:
        if not tokens:
            return {}, {}, 0.0

        pron_counts = {
            "1p_sing": 0,
            "1p_pl": 0,
            "2p_sing_informal": 0,
            "2p_formal_plural": 0,
            "3p_sing": 0,
            "3p_pl": 0,
            "self_pronoun": 0,
            "poss_Apro": 0,
            "poss_Poss": 0,
        }
        for token in tokens:
            parse = self.morph.parse(token)[0]
            pos = parse.tag.POS
            person = parse.tag.person
            number = parse.tag.number

            if pos == "NPRO":
                if person == "1per":
                    if number == "sing":
                        pron_counts["1p_sing"] += 1
                    elif number == "plur":
                        pron_counts["1p_pl"] += 1
                elif person == "2per":
                    if number == "sing":
                        pron_counts["2p_sing_informal"] += 1
                    elif number == "plur":
                        pron_counts["2p_formal_plural"] += 1
                elif person == "3per":
                    if number == "sing":
                        pron_counts["3p_sing"] += 1
                    elif number == "plur":
                        pron_counts["3p_pl"] += 1
                else:
                    pron_counts["self_pronoun"] += 1

            if pos in ["ADJF", "ADJS"]:
                if "Apro" in parse.tag:
                    pron_counts["poss_Apro"] += 1
                # elif 'Poss' in parse.tag:
                #     pron_counts["poss_Poss"] += 1

        freq = sum(pron_counts.values()) / cnt_tokens_important
        pron_rel = {k: v / cnt_tokens_important for k, v in pron_counts.items()}
        return pron_counts, pron_rel, freq

    def export_detect_time_consistency(
        self, doc: Doc
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        verbs = {"Pres": 0, "Past": 0, "Fut": 0}

        for token in doc.tokens:
            if "Tense" in token.feats:
                tense = token.feats["Tense"]
                if tense in verbs:
                    verbs[tense] += 1

        total = sum(verbs.values()) or 1
        rel = {k: v / total for k, v in verbs.items()}
        return verbs, rel

    def export_unimportant_pos(
        self, tokens: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, float], float]:
        """
        Подсчитывает количество "неважных" частей речи в тексте
        (например, предлогов и союзов) и их долю среди всех токенов.
        """
        unimportant_pos_tags = {
            "PREP": 0,
            "CONJ": 0,
            "PRCL": 0,
            "INTJ": 0,
            "PRED": 0,
            "COMP": 0,
        }
        if not tokens:
            return unimportant_pos_tags, unimportant_pos_tags

        for token in tokens:
            parse = self.morph.parse(token)[0]
            if parse.tag.POS in unimportant_pos_tags:
                unimportant_pos_tags[parse.tag.POS] += 1

        ration_tag = {k: v / len(tokens) for k, v in unimportant_pos_tags.items()}
        ratio = sum(unimportant_pos_tags.values()) / len(tokens)
        return unimportant_pos_tags, ration_tag, ratio

    def export_emotion_detection(
        self,
        sents: List[str],
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        emotion_dict = {
            "joy": 0,
            "interest": 0,
            "surpise": 0,
            "sadness": 0,
            "anger": 0,
            "disgust": 0,
            "fear": 0,
            "guilt": 0,
            "neutral": 0,
        }
        for sent in sents:
            emotions_detect = self.pipe_emotions(sent)
            for emotion_value in emotions_detect:
                emotion_dict[emotion_value["label"]] += emotion_value["score"]

        total = sum(emotion_dict.values()) or 1
        rel = {k: v / total for k, v in emotion_dict.items()}
        return emotion_dict, rel

    def export_neologism_or_strange_words(
        self, tokens: List[str], threshold: int = 12
    ) -> Tuple[int, float]:
        if not tokens:
            return 0, 0.0
        freq_good = 0
        for token in tokens:
            norm = self.morph.parse(token)[0].normal_form
            if norm in self.freq_words:
                if self.freq_words[norm] > threshold:
                    freq_good += 1

        rare_ratio = 1 - (freq_good / len(tokens))
        return (len(tokens) - freq_good), rare_ratio

    def export_banned_words(self, tokens: List[str]) -> Tuple[int, float]:
        if not tokens:
            return 0, 0.0
        found = [
            t for t in tokens if self.morph.parse(t)[0].normal_form in self.banned_words
        ]
        ratio = len(found) / len(tokens)
        return len(found), ratio

    def export_cohesive_markers_count(
        self,
        sentences_tokens: List[List[str]],
    ) -> Tuple[int, float]:
        if not sentences_tokens:
            return 0, 0.0

        freq = 0
        for tokens in sentences_tokens:
            c = self._find_conjunction_types(
                tokens, {"cohesive": self.conjunctions_set}
            )
            if c:
                freq += 1
        return freq, freq / len(sentences_tokens)

    def export_coherence_and_paraphrasing(
        self, sentences_embeddings: np.ndarray, context_size=5
    ) -> Tuple[float, float, float]:
        """
        Строит матрицу связности, где каждое предложение оценивается
        относительно предыдущего контекста.
        """
        len_sentences = len(sentences_embeddings)

        # Создаем матрицу связности
        context_matrix = np.zeros((len_sentences, context_size))
        context_abs_matrix = np.zeros((len_sentences, context_size))
        # Создаем матрицу перефразировки и повторения
        repetition_and_paraphrasing_matrix = np.zeros((len_sentences, context_size))

        # Вычисляем косинусное сходство между текущим и j-предыдущим предложением
        for i in range(1, len_sentences):
            for j in range(1, min(i + 1, context_size + 1)):
                similarity = cosine_similarity(
                    sentences_embeddings[i].reshape(1, -1),
                    sentences_embeddings[i - j].reshape(1, -1),
                )[0][0]
                context_matrix[i, j - 1] = similarity
                context_abs_matrix[i, j - 1] = 1 if similarity > 0.3 else EPS
                repetition_and_paraphrasing_matrix[i, j - 1] = (
                    1 if similarity > 0.7 else EPS
                )

        # Вычисляем общую оценку связности
        # Исключаем нулевые элементы при расчете среднего
        non_zero_elements = context_matrix[context_matrix > 0]
        overall_coherence = (
            np.mean(non_zero_elements) if len(non_zero_elements) > 0 else 0.0
        )

        # Вычисляем общую оценку связности по абсолютным значениям
        # Исключаем нулевые элементы при расчете среднего
        non_zero_abs_elements = context_abs_matrix[context_abs_matrix > 0]
        overall_abs_coherence = (
            np.mean(non_zero_abs_elements) if len(non_zero_abs_elements) > 0 else 0.0
        )

        # Вычисляем общую оценку перефраза
        # Исключаем нулевые элементы при расчете среднего
        non_zero_lexical_repetition_and_paraphrasing = (
            repetition_and_paraphrasing_matrix[repetition_and_paraphrasing_matrix > 0]
        )
        overall_lexical_repetition_and_paraphrasing = (
            np.mean(non_zero_lexical_repetition_and_paraphrasing)
            if len(non_zero_lexical_repetition_and_paraphrasing) > 0
            else 0.0
        )

        return (
            overall_coherence,
            overall_abs_coherence,
            overall_lexical_repetition_and_paraphrasing,
        )

    def export_refference(self, doc: Doc, morph_vocab: MorphVocab) -> Tuple[int, float]:
        for t in doc.tokens:
            t.lemmatize(morph_vocab)

        mentions = []
        for sent in doc.sents:
            for token in sent.tokens:
                emb_w = self._get_word_embedding(token.text, sent.text)
                mentions.append(
                    {
                        "text": token.text,
                        "lemma": token.lemma.lower(),
                        "pos": token.pos,
                        "gender": token.feats.get("Gender"),
                        "number": token.feats.get("Number"),
                        "person": token.feats.get("Person"),
                        "embedding": emb_w,
                    }
                )

        last_mentions = []
        results = []

        demonstratives = [
            "оное",
            "оный",
            "сей",
            "сие",
            "столько",
            "таковой",
            "такой",
            "тем",
            "то",
            "тот",
            "этакий",
            "это",
            "этот",
        ]
        possessives_map = {
            "мой": ("1", "Sing"),
            "твой": ("2", "Sing"),
            "наш": ("1", "Plur"),
            "ваш": ("2", "Plur"),
            "их": ("3", "Plur"),
            "его": ("3", "Sing"),
            "её": ("3", "Sing"),
            "свой": None,
        }

        def cos_sim(x, y):
            denom = (np.linalg.norm(x) * np.linalg.norm(y)) + EPS
            return float(np.dot(x, y) / denom)

        for m in mentions:
            lemma_m = m["lemma"]
            pos_m = m["pos"]
            per_m = m["person"]
            gen_m = m["gender"]
            num_m = m["number"]
            emb_m = m["embedding"]

            if pos_m in ("PRON", "NPRO") and per_m == "3":
                # Поиск кандидатов
                cand = []
                for pm in last_mentions:
                    if pm["gender"] == gen_m and pm["number"] == num_m:
                        score = cos_sim(emb_m, pm["embedding"])
                        cand.append(score)
                if cand:
                    best_s = max(cand)
                    results.append(best_s)
            elif lemma_m in demonstratives:
                cand = [cos_sim(emb_m, pm["embedding"]) for pm in last_mentions]
                if cand:
                    best_s = max(cand)
                    results.append(best_s)
            elif lemma_m in possessives_map:
                pinfo = possessives_map[lemma_m]
                if pinfo is not None:
                    p_person, p_number = pinfo
                    cand = []
                    for pm in last_mentions:
                        if pm["person"] == p_person and pm["number"] == p_number:
                            score = cos_sim(emb_m, pm["embedding"])
                            cand.append(score)
                    if cand:
                        best_s = max(cand)
                        results.append(best_s)
            else:
                last_mentions.append(m)

        if not results:
            return 0, 0.0
        threshold = 0.33
        count_ref = sum(1 for r in results if r > threshold)
        return count_ref, count_ref / len(results)

    def export_analyze_syntax_complexity(
        self,
        sentences: List[str],
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        sents = [s.strip().lower() for s in sentences]
        if not sents:
            return {}, {}

        results = {
            "simple": 0,
            "asyndetic": 0,
            "compound_coord": 0,
            "compound_subord": 0,
            "linkers": 0,
            "multiple_connections": 0,
            # "undefined": 0
        }
        total_count = 0
        for sent_text in sents:
            if not sent_text:
                continue
            doc = self.stanza_nlp(sent_text)
            if not doc.sentences:
                # results['undefined'] += 1
                total_count += 1
                continue

            s0 = doc.sentences[0]
            tokens_info = []
            nsubj_count = 0
            for w in s0.words:
                tokens_info.append(
                    {
                        "lemma": w.lemma.lower() if w.lemma else w.text.lower(),
                        "deprel": w.deprel,
                    }
                )
                if w.deprel == "nsubj":
                    nsubj_count += 1

            tokens = [t["lemma"] for t in tokens_info]
            deprels = [t["deprel"] for t in tokens_info]

            coord = (
                self._find_conjunction_types(tokens, self.coordinating_conjunctions)
                or []
            )
            subord = (
                self._find_conjunction_types(tokens, self.subodinating_conjunctions)
                or []
            )
            linkers = (
                self._find_conjunction_types(tokens, {"link": self.other_linkers}) or []
            )

            hasCoord = (len(coord) != 0) and any(
                rel in ("conj", "cc", "parataxis") for rel in deprels
            )
            hasSubord = (len(subord) != 0) and any(
                rel in ("mark", "advcl", "acl", "csubj", "ccomp") for rel in deprels
            )
            hasLinkers = (len(linkers) != 0) and any(
                rel in ("mark", "discourse") for rel in deprels
            )

            if hasCoord:
                results["compound_coord"] += 1
            if hasSubord:
                results["compound_subord"] += 1
            if hasLinkers:
                results["linkers"] += 1
            if nsubj_count > 1 and not hasCoord and not hasSubord:
                results["asyndetic"] += 1
            if nsubj_count == 1 and not hasCoord and not hasSubord:
                results["simple"] += 1
            if hasCoord and hasSubord:
                results["multiple_connections"] += 1

            total_count += 1

        if total_count == 0:
            return results, {k: 0.0 for k in results}
        rel_res = {k: v / total_count for k, v in results.items()}
        return results, rel_res

    def export_analyze_descriptive_structures(
        self, sentences: List[str], cnt_tokens_important: int
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        descriptive_struct = {
            "adjectives": 0,
            "adverbs": 0,
            "participles": 0,
            "gerunds": 0,
            "enumeration_groups": 0,
        }
        for sent_text in sentences:
            st = sent_text.strip().lower()
            if not st:
                continue
            doc = self.stanza_nlp(st)
            if not doc.sentences:
                continue
            s0 = doc.sentences[0]
            for w in s0.words:
                upos = w.upos
                feats = w.feats or ""
                if upos == "ADJ":
                    descriptive_struct["adjectives"] += 1
                elif upos == "ADV":
                    descriptive_struct["adverbs"] += 1
                elif upos == "VERB":
                    if "VerbForm=Part" in feats:
                        descriptive_struct["participles"] += 1
                    elif "VerbForm=Conv" in feats:
                        descriptive_struct["gerunds"] += 1

            nodes = [w.id for w in s0.words]
            conj_graph = {node: set() for node in nodes}
            for w in s0.words:
                if w.deprel == "conj":
                    head_id = w.head
                    child_id = w.id
                    if head_id in conj_graph:
                        conj_graph[head_id].add(child_id)
                    if child_id in conj_graph:
                        conj_graph[child_id].add(head_id)

            visited = set()
            groups_in_sentence = 0

            def dfs(start):
                stack = [start]
                component = []
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        component.append(node)
                        for neigh in conj_graph[node]:
                            if neigh not in visited:
                                stack.append(neigh)
                return component

            for node in conj_graph:
                if node not in visited and conj_graph[node]:
                    comp = dfs(node)
                    if len(comp) >= 2:
                        groups_in_sentence += 1

            descriptive_struct["enumeration_groups"] += groups_in_sentence

        rel_struct = {
            k: v / cnt_tokens_important for k, v in descriptive_struct.items()
        }
        return descriptive_struct, rel_struct

    def export_analyze_modals_in_text(
        self, sentences_tokens: List[List[str]], sentences_emdbeddings: np.ndarray
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        counts = {"epistemic": 0, "deontic": 0, "inclusive": 0, "not_modal": 0}

        if len(sentences_tokens) == 0:
            return counts, counts

        cnt_verb = 0
        for i, sentence in enumerate(sentences_tokens):
            for token in sentence:
                tag = self.morph.parse(token)[0].tag
                normal_form = self.morph.parse(token)[0].normal_form
                if "VERB" in tag or "INFN" in tag or "GRND" in tag or "PRTF" in tag:
                    label = "not_modal"
                    # Проверяем, является ли это слово потенциально модальным
                    cnt_verb += 1
                    if normal_form in self.all_modal_verbs:
                        pred = self.catboost_modalities.predict(
                            sentences_emdbeddings[i]
                        )
                        label = pred[0]
                        while type(label) in [list, np.ndarray]:
                            label = label[0]

                    if label in counts:
                        counts[label] += 1
        if cnt_verb == 0:
            return counts, counts
        rel_counts = {k: v / cnt_verb for k, v in counts.items()}
        return counts, rel_counts

    def export_ner_entities(self, doc: Doc) -> Tuple[Dict[str, int], Dict[str, float]]:
        ner_tags = {"PER": [], "ORG": [], "LOC": []}

        if not doc.spans:
            return {k: 0 for k in ner_tags.keys()}, {k: 0.0 for k in ner_tags.keys()}

        for span in doc.spans:
            ner_tags[span.type].append(
                re.findall(r"[\w|-]+", span.text.strip().lower(), re.UNICODE)
            )

        tokens_count = len(re.findall(r"[\w|-]+", doc.text.strip().lower(), re.UNICODE))
        tokens_ner = [
            len(entity) for entities in ner_tags.values() for entity in entities
        ]
        norm_count = tokens_count - sum(tokens_ner) + len(tokens_ner)

        abs_ner_tags = {tag: len(entities) for tag, entities in ner_tags.items()}
        norm_ner_tags = {
            tag: len(entities) / norm_count for tag, entities in ner_tags.items()
        }

        return abs_ner_tags, norm_ner_tags

    def export_verb_clusters(
        self,
        tokens: List[str],
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        verb_tokens = []
        for token in tokens:
            tag = self.morph.parse(token)[0].tag
            if "VERB" in tag or "INFN" in tag or "GRND" in tag or "PRTF" in tag:
                verb_tokens.append(self.morph.parse(token)[0].normal_form)

        cnt_verbs = len(verb_tokens)
        clusters_verbs = {cluster: [] for cluster in self.verbs_clusters.keys()}
        for verb_token in verb_tokens:
            find = False
            for cluster, verbs_list in self.verbs_clusters.items():
                if find:
                    continue
                elif verb_token in verbs_list:
                    clusters_verbs[cluster].append(verb_token)
                    find = True
                    break

        cnt_clusters_verbs = {f"cnt_{k}": len(v) for k, v in clusters_verbs.items()}
        ration_clusters_verbs = {
            f"ratio_{k}": len(v) / cnt_verbs if cnt_verbs > 0 else 0
            for k, v in clusters_verbs.items()
        }

        return cnt_clusters_verbs, ration_clusters_verbs

    def export_ne_verb_ngrams(self, tokens: List[str]) -> Tuple[int, float]:
        results = []
        for i in range(len(tokens) - 1):
            if tokens[i] == "не":
                start = i + 1
                end = i + 4 if len(tokens) - i > 4 else len(tokens)
                parses = [self.morph.parse(tokens[j]) for j in range(start, end)]
                n_gramms = ["не"]
                for j, parse in enumerate(parses):
                    for p in parse:
                        if "VERB" in p.tag or "INFN" in p.tag:
                            n_gramms.append(tokens[i + j + 1])
                        else:
                            break
                results.append(n_gramms)
        negative_energy = len([gramm for n_gramms in results for gramm in n_gramms])
        ratio_negative_energy = negative_energy / len(tokens) if len(tokens) > 0 else 0
        return negative_energy, ratio_negative_energy

    def export_negative_particles(self, tokens: List[str]) -> Tuple[int, float]:
        cnt_tokens = len(tokens)
        cnt_negative_particles = len(
            self._find_conjunction_types(
                tokens,
                relation={"negative_particles": self.negative_particles},
                return_set=False,
            )
        )
        ratio_negative_particles = (
            cnt_negative_particles / cnt_tokens if cnt_tokens > 0 else 0
        )
        return cnt_negative_particles, ratio_negative_particles

    def export_semantic_clusters(
        self,
        tokens: List[str],
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        cnt_tokens = len(tokens)

        clusters_lexemes = self._find_conjunction_types(
            tokens,
            relation=self.semantic_clusters,
            return_set=False,
            return_dict=True,
            use_morph=True,
        )

        cnt_clusters_lexemes = {f"cnt_{k}": len(v) for k, v in clusters_lexemes.items()}
        ration_clusters_lexemes = {
            f"ratio_{k}": len(v) / cnt_tokens if cnt_tokens > 0 else 0
            for k, v in clusters_lexemes.items()
        }

        return cnt_clusters_lexemes, ration_clusters_lexemes

    # ===========================================================================
    #                            Основная логика анализа
    # ===========================================================================
    def analyze_text(self, input_data: str | dict) -> Dict[str, Any]:
        """Извлекает признаки из текста."""
        if isinstance(input_data, dict):
            text_primary = input_data["text"]
            segments = input_data["chunks"]
        elif isinstance(input_data, dict):
            text_primary = input_data

        tokens_primary = re.findall(r"[\w|-]+", text_primary.lower(), re.UNICODE)
        text = self._normalize_text(text_primary)
        tokens = re.findall(r"[\w|-]+", text.lower(), re.UNICODE)
        sents = re.split(r"(?<=[.!?]) +", text)
        sents_tokens = [
            re.findall(r"[\w|-]+", s.lower(), re.UNICODE) for s in sents if s.strip()
        ]
        sents_embs = self._get_embeddings(sents)
        tokens_important, cnt_tokens_important = self._get_token_important(tokens)

        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)

        # Признаки
        features: Dict[str, Any] = {}
        features["text"] = text
        if segments:
            features["speed_speech"] = self.export_speed_speech(segments)
        features["mean_sentence_length"] = self.export_mean_sentence_length(
            sents_tokens
        )
        features["mean_word_len"] = self.export_mean_word_len(tokens)
        features["count_unintelligible_words"] = self.export_count_unintelligible_words(
            tokens_primary, tokens
        )
        features["type_token_ratio"] = self.export_type_token_ratio(tokens)
        features["classify_pronouns"] = self.export_classify_pronouns(
            tokens, cnt_tokens_important
        )
        features["detect_time_consistency"] = self.export_detect_time_consistency(doc)
        features["unimportant_pos"] = self.export_unimportant_pos(tokens)
        features["emotion_detection"] = self.export_emotion_detection(sents)
        features["neologism_or_strange_words"] = self.export_neologism_or_strange_words(
            tokens_important
        )
        features["banned_words"] = self.export_banned_words(tokens_important)
        features["cohesive_markers_count"] = self.export_cohesive_markers_count(
            sents_tokens
        )
        features["coherence_and_paraphrasing"] = self.export_coherence_and_paraphrasing(
            sents_embs
        )
        features["refference"] = self.export_refference(doc, self.morph_vocab)
        features["syntax_complexity"] = self.export_analyze_syntax_complexity(sents)
        features["descriptive_structures"] = self.export_analyze_descriptive_structures(
            sents, cnt_tokens_important
        )
        features["modals_in_text"] = self.export_analyze_modals_in_text(
            sents_tokens, sents_embs
        )
        features["ner_entities"] = self.export_ner_entities(doc)
        features["verb_clusters"] = self.export_verb_clusters(tokens)
        features["ne_verb_ngrams"] = self.export_ne_verb_ngrams(tokens)
        features["negative_particles"] = self.export_negative_particles(tokens)
        features["semantic_clusters"] = self.export_semantic_clusters(tokens)

        return features


# ===========================================================================
#                            LinguisticExtractor
# ===========================================================================
class LinguisticExtractor(BaseExtractor):
    """
    Унифицированный интерфейс для лингвистического анализа речи с режимами:
      • analysis_mode='non_cumulative' — окна [t, t+step_size]
      • analysis_mode='cumulative'     — окна [0, t]
      • analysis_mode='full'           — один кадр на весь файл [0, T]

    Возвращает DataFrame с колонками: start_sec, end_sec, признаки
    """

    def __init__(
        self,
        resources: ResourceLoader,
        analysis_mode: Literal["cumulative", "non_cumulative", "full"] = "cumulative",
        sample_rate: int = 16000,
        window_size: float = 5.0,
        step_size: float = 2.0,
    ):
        super().__init__(sample_rate, window_size, step_size)
        self.analysis_mode = analysis_mode
        self.extractor = FeatureExtractor(resources)
        self.words_per_sec = self.extractor.words_per_sec

    # ===========================================================================
    #                               Вспомогательные
    # ===========================================================================
    @staticmethod
    def _flatten(feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for k, v in feature_dict.items():
            len_v = len(v)
            if isinstance(v, str):
                flat[f"{k}_str"] = v
            elif len_v == 2 and isinstance(v[0], float) and isinstance(v[1], float):
                flat[f"{k}_mean"] = v[0]
                flat[f"{k}_ratio"] = v[1]
            elif len_v == 2 and isinstance(v[0], int) and isinstance(v[1], float):
                flat[f"{k}_cnt"] = v[0]
                flat[f"{k}_ratio"] = v[1]
            elif len_v == 2 and isinstance(v[0], dict) and isinstance(v[1], dict):
                for kk, vv in v[0].items():
                    flat[f"{k}__{kk}_cnt"] = vv
                for kk, vv in v[1].items():
                    flat[f"{k}__{kk}_ratio"] = vv
            elif (
                len_v == 3
                and isinstance(v[0], float)
                and isinstance(v[1], float)
                and isinstance(v[2], float)
            ):
                flat[f"{k}_mean"] = v[0]
                flat[f"{k}_cnt_mean"] = v[1]
                flat[f"{k}_ratio"] = v[2]
            elif (
                len_v == 3
                and isinstance(v[0], int)
                and isinstance(v[1], float)
                and isinstance(v[2], float)
            ):
                flat[f"{k}_cnt"] = v[0]
                flat[f"{k}_mean"] = v[1]
                flat[f"{k}_ratio"] = v[2]
            elif (
                len_v == 3
                and isinstance(v[0], int)
                and isinstance(v[1], int)
                and isinstance(v[2], float)
            ):
                flat[f"{k}_len_first"] = v[0]
                flat[f"{k}_len_second"] = v[1]
                flat[f"{k}_ratio"] = v[2]
            elif (
                len_v == 3
                and isinstance(v[0], dict)
                and isinstance(v[1], dict)
                and isinstance(v[2], float)
            ):
                for kk, vv in v[0].items():
                    flat[f"{k}__{kk}_cnt"] = vv
                for kk, vv in v[1].items():
                    flat[f"{k}__{kk}_ratio"] = vv
                flat[f"{k}_ratio"] = v[2]
            else:
                print(k, type(v))
        return flat

    def _audio_duration_sec(self, path: Path) -> float:
        try:
            import soundfile as _sf

            info = _sf.info(str(path))
            return float(info.frames) / float(info.samplerate or self.sample_rate)
        except Exception:
            y, sr = librosa.load(str(path), sr=None, mono=True)
            return len(y) / float(sr or self.sample_rate)

    def _text_duration_sec(self, text: str) -> float:
        words = self.extractor._tokenize(text)
        return len(words) / self.words_per_sec if words else 0.0

    def _analyze_audio_full(self, path: Path) -> pd.DataFrame:
        transcribe = self.extractor._transcribe_full(path)
        feats = self.extractor.analyze_text(transcribe)
        row = self._flatten(feats) | {
            "start_sec": 0.0,
            "end_sec": self._audio_duration_sec(path),
        }
        return pd.DataFrame([row])

    def _analyze_audio_cumulative(self, path: Path) -> pd.DataFrame:
        """
        Анализирует текст кумулятивно.
        """
        y, sr = self.extractor._load_audio_mono(str(path), target_sr=self.sample_rate)
        rows: List[Dict[str, Any]] = []
        for start, end in self.iter_windows(y, sr):
            seg = y[0:end]
            t0, t1 = start / sr, end / sr
            transcribe = self.extractor._transcribe_segment(seg, sr)
            feats = self.extractor.analyze_text(transcribe)
            rows.append(
                self._flatten(feats) | {"start_sec": float(t0), "end_sec": float(t1)}
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _analyze_audio_non_cumulative(self, path: Path) -> pd.DataFrame:
        """
        Анализирует текст не кумулятивно.
        """
        y, sr = self.extractor._load_audio_mono(path, target_sr=self.sample_rate)
        rows: List[Dict[str, Any]] = []
        for start, end in self.iter_windows(y, sr):
            seg = y[start:end]
            t0, t1 = start / sr, end / sr
            transcribe = self.extractor._transcribe_segment(seg, sr)
            feats = self.extractor.analyze_text(transcribe)
            rows.append(
                self._flatten(feats) | {"start_sec": float(t0), "end_sec": float(t1)}
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ===========================================================================
    #                          Извлечение признаков
    # ===========================================================================

    def extract_from_path(self, path: str | Path) -> pd.DataFrame:
        """
        Анализирует аудио по входному пути до файла.
        """
        p = Path(path)
        suf = p.suffix.lower()
        if suf in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
            if self.analysis_mode == "full":
                return self._analyze_audio_full(p)
            elif self.analysis_mode == "cumulative":
                return self._analyze_audio_cumulative(p)
            elif self.analysis_mode == "non_cumulative":
                return self._analyze_audio_non_cumulative(p)
            else:
                raise ValueError(f"Unknown analysis_mode: {self.analysis_mode}")
        elif suf == ".txt":
            text = p.read_text(encoding="utf-8")
            return self.extract_from_text(text)
        else:
            raise ValueError(f"Unsupported file format: {suf}")

    def extract_from_array(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        """
        Анализирует по входному массиву и частоте дискретизации.
        """
        tmp = Path(f".tmp_audio_{hashlib.md5(y.tobytes()).hexdigest()[:10]}.wav")
        sf.write(str(tmp), y, sr)
        try:
            return self.extract_from_path(tmp)
        finally:
            try:
                tmp.unlink()
            except Exception:
                pass

    def extract_from_text(self, text: str) -> pd.DataFrame:
        """
        Та же логика режимов для текста. Время моделируется: 2 слова = 1 сек.
        """
        if not text:
            if self.analysis_mode == "full":
                feats = self.extractor.analyze_text(text)
                return pd.DataFrame(
                    [self._flatten(feats) | {"start_sec": 0.0, "end_sec": 0.0}]
                )
            return pd.DataFrame()

        text_norm = self.extractor._normalize_text(text)
        words = self.extractor._tokenize(text_norm)

        duration = len(words) / self.words_per_sec

        if self.analysis_mode == "full":
            feats = self.extractor.analyze_text(text_norm)
            row = self._flatten(feats) | {"start_sec": 0.0, "end_sec": float(duration)}
            return pd.DataFrame([row])

        rows: List[Dict[str, Any]] = []
        if self.analysis_mode == "cumulative":
            # накопительные окна: берем первые K слов для текущего t
            t = self.window_size
            while t <= max(duration, self.window_size) + 1e-9:
                num_words = int(round(t * self.words_per_sec))
                slice_text = " ".join(words[: max(1, min(num_words, len(words)))])
                feats = self.extractor.analyze_text(slice_text)
                rows.append(
                    self._flatten(feats) | {"start_sec": 0.0, "end_sec": float(t)}
                )
                t += self.step_size
                if t > duration and t - self.step_size >= duration:
                    break
        elif self.analysis_mode == "non_cumulative":
            # обычные окна текста
            win_words = int(round(self.window_size * self.words_per_sec))
            step_words = int(round(self.step_size * self.words_per_sec))
            if win_words <= 0:
                win_words = max(1, int(self.words_per_sec * 1))  # fallback: 1 сек
            if step_words <= 0:
                step_words = win_words

            for start in range(0, len(words), step_words):
                end = start + win_words
                if start >= len(words):
                    break
                slice_text = " ".join(words[start : min(end, len(words))])
                feats = self.extractor.analyze_text(slice_text)
                rows.append(
                    self._flatten(feats)
                    | {
                        "start_sec": float(start / self.words_per_sec),
                        "end_sec": float(min(end, len(words)) / self.words_per_sec),
                    }
                )
                if end >= len(words):
                    break
        else:
            raise ValueError(f"Unknown analysis_mode: {self.analysis_mode}")

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def extract_dataset(
        self,
        input_dir: str | Path,
        output_csv: str | Path,
        verbose: bool = True,
    ) -> None:
        """
        Обрабатывает директорию файлов (аудио или .txt) по заданному режиму при инициализации LinguisticExtractor.
        Параметры window_size/step_size и analysis_mode применяются к каждому файлу.
        Принимает на вход путь до директории с анализируемыми файлами и путь, куда сложить результат (формат .csv указать вручную).
        """
        input_dir = Path(input_dir)
        output_csv = Path(output_csv)

        files = sorted(
            [
                p
                for p in Path(input_dir).glob("*")
                if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".txt")
            ]
        )
        all_frames: List[pd.DataFrame] = []
        iterator = files
        if verbose:
            try:
                from tqdm.auto import tqdm  # type: ignore

                iterator = tqdm(files, desc="Processing")
            except Exception:
                pass

        for path in iterator:
            try:
                df = self.extract_from_path(path)
                if not df.empty:
                    df.insert(0, "file", path.name)
                    all_frames.append(df)
            except Exception as e:  # pragma: no cover
                print(f"⚠️ Error processing {path.name}: {e}")

        if not all_frames:
            raise RuntimeError(
                "Не удалось извлечь лингвистические признаки ни из одного файла."
            )

        result = pd.concat(all_frames, ignore_index=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        if verbose:
            print(f"✅ Сохранено {len(result)} окон в {output_csv}")
