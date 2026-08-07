"""RapidOCR PP-OCRv6 tiny ONNX backend for low-latency local OCR."""

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RAPIDOCR_MODEL_CODE = "rapidocr"
RAPIDOCR_DISPLAY_NAME = "RapidOCR PP-OCRv6 tiny (ONNX)"
RAPIDOCR_PACKAGE_VERSION = "3.9.2"
RAPIDOCR_ONNXRUNTIME_VERSION = "1.28.0"
RAPIDOCR_INTRA_OP_THREADS = 2
RAPIDOCR_INTER_OP_THREADS = 1
RAPIDOCR_MIN_SCORE = 0.45


class RapidOCRUnavailableError(RuntimeError):
    """Raised when the configured RapidOCR runtime cannot be initialized."""


@dataclass(frozen=True)
class RapidOCRSettings:
    ocr_version: str = "PP-OCRv6"
    model_size: str = "tiny"
    engine: str = "onnxruntime"
    device: str = "cpu"
    intra_op_num_threads: int = RAPIDOCR_INTRA_OP_THREADS
    inter_op_num_threads: int = RAPIDOCR_INTER_OP_THREADS
    min_score: float = RAPIDOCR_MIN_SCORE
    text_det_limit_side_len: int = 960
    text_det_limit_type: str = "max"


@dataclass(frozen=True)
class RapidOCRLine:
    text: str
    confidence: float
    bbox: object = None


_ENGINE = None
_ENGINE_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.Lock()


def _model_root_dir():
    override = str(os.environ.get("OCR_TRANSLATOR_RAPIDOCR_MODEL_DIR", "") or "").strip()
    if override:
        root = Path(override).expanduser()
    else:
        local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
        if local_app_data:
            root = Path(local_app_data) / "OCR-Translator" / "rapidocr" / "models"
        else:
            root = Path.home() / ".ocr-translator" / "rapidocr" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_engine():
    try:
        from rapidocr import (
            LangDet,
            LangRec,
            ModelType,
            OCRVersion,
            RapidOCR,
        )
    except Exception as error:
        raise RapidOCRUnavailableError(
            "RapidOCR 3.9.2 and ONNX Runtime 1.28.0 are required"
        ) from error

    params = {
        "Global.model_root_dir": str(_model_root_dir()),
        "Global.use_cls": False,
        "Global.text_score": RAPIDOCR_MIN_SCORE,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.intra_op_num_threads": RAPIDOCR_INTRA_OP_THREADS,
        "EngineConfig.onnxruntime.inter_op_num_threads": RAPIDOCR_INTER_OP_THREADS,
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Det.lang_type": LangDet.EN,
        "Det.model_type": ModelType.TINY,
        "Det.limit_side_len": 960,
        "Det.limit_type": "max",
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.lang_type": LangRec.EN,
        "Rec.model_type": ModelType.TINY,
    }
    try:
        return RapidOCR(params=params)
    except Exception as error:
        raise RapidOCRUnavailableError(
            "RapidOCR PP-OCRv6 tiny ONNX initialization failed"
        ) from error


def get_rapidocr_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = _build_engine()
        return _ENGINE


def clear_rapidocr_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = None


def prepare_rapidocr_image(pil_image, settings=None):
    if pil_image is None:
        raise ValueError("RapidOCR image is missing")
    return pil_image.convert("RGB")


def _box_sort_key(box):
    try:
        points = list(box)
        return (
            min(float(point[1]) for point in points),
            min(float(point[0]) for point in points),
        )
    except Exception:
        return (0.0, 0.0)


def flatten_rapidocr_result(result, keep_linebreaks=False, min_score=RAPIDOCR_MIN_SCORE):
    raw_texts = getattr(result, "txts", None)
    raw_scores = getattr(result, "scores", None)
    raw_boxes = getattr(result, "boxes", None)
    texts = tuple(raw_texts) if raw_texts is not None else ()
    scores = tuple(raw_scores) if raw_scores is not None else ()
    boxes = tuple(raw_boxes) if raw_boxes is not None else ()
    items = []
    for index, text in enumerate(texts):
        normalized_text = str(text or "").strip()
        if not normalized_text:
            continue
        try:
            score = float(scores[index])
        except (IndexError, TypeError, ValueError):
            score = 0.0
        if score < float(min_score):
            continue
        bbox = boxes[index] if index < len(boxes) else None
        items.append((_box_sort_key(bbox), normalized_text, score, bbox))
    items.sort(key=lambda item: item[0])
    lines = [
        RapidOCRLine(text=text, confidence=score, bbox=bbox)
        for _sort_key, text, score, bbox in items
    ]
    separator = "\n" if keep_linebreaks else " "
    return separator.join(line.text for line in lines), lines


def recognize_with_rapidocr(pil_image, settings=None, keep_linebreaks=False):
    prepared = prepare_rapidocr_image(pil_image, settings)
    engine = get_rapidocr_engine()
    with _INFERENCE_LOCK:
        result = engine(
            np.asarray(prepared),
            use_det=True,
            use_cls=False,
            use_rec=True,
            text_score=RAPIDOCR_MIN_SCORE,
        )
    return flatten_rapidocr_result(
        result,
        keep_linebreaks=keep_linebreaks,
        min_score=RAPIDOCR_MIN_SCORE,
    )


def summarize_rapidocr_settings(settings=None):
    settings = settings or RapidOCRSettings()
    return (
        f"version={settings.ocr_version} size={settings.model_size} "
        f"engine={settings.engine} device={settings.device} "
        f"threads={settings.intra_op_num_threads}/{settings.inter_op_num_threads} "
        f"min_score={settings.min_score}"
    )
