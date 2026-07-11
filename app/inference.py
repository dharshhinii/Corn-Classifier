from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import ModelConfig


class ImageClassifier:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Any = None
        self.interpreter: Any = None
        self.input_details = None
        self.output_details = None
        self.lock = threading.Lock()

    def load(self) -> None:
        if not self.config.path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.config.path}"
            )

        import tensorflow as tf

        if self.config.model_type == "keras":
            self.model = tf.keras.models.load_model(
                self.config.path,
                compile=False,
            )
        elif self.config.model_type == "tflite":
            self.interpreter = tf.lite.Interpreter(
                model_path=str(self.config.path)
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]
        else:
            raise ValueError(
                "model type must be 'keras' or 'tflite'"
            )

    def _prepare(self, image_path: str) -> np.ndarray:
        path = Path(image_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    self.config.image_size,
                    Image.Resampling.BILINEAR,
                )
                array = np.asarray(image, dtype=np.float32)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Invalid image: {path}") from exc

        return np.expand_dims(array, axis=0)

    def _keras_predict(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model(batch, training=False)
        )[0].astype(np.float32)

    def _tflite_predict(self, batch: np.ndarray) -> np.ndarray:
        input_dtype = self.input_details["dtype"]
        scale, zero_point = self.input_details["quantization"]

        if np.issubdtype(input_dtype, np.integer):
            if scale == 0:
                raise ValueError("Invalid TFLite input scale")
            batch = np.round(batch / scale + zero_point)
            limits = np.iinfo(input_dtype)
            batch = np.clip(
                batch, limits.min, limits.max
            ).astype(input_dtype)
        else:
            batch = batch.astype(input_dtype)

        with self.lock:
            self.interpreter.set_tensor(
                self.input_details["index"], batch
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(
                self.output_details["index"]
            )[0]

        output_scale, output_zero = self.output_details["quantization"]
        if np.issubdtype(output.dtype, np.integer) and output_scale:
            output = output_scale * (
                output.astype(np.float32) - output_zero
            )

        return output.astype(np.float32)

    @staticmethod
    def _probabilities(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)

        already_probabilities = (
            np.all(values >= 0)
            and np.all(values <= 1)
            and np.isclose(values.sum(), 1.0, atol=1e-3)
        )

        if already_probabilities:
            return values.astype(np.float32)

        values = values - np.max(values)
        exp_values = np.exp(values)
        return (exp_values / exp_values.sum()).astype(np.float32)

    def predict(self, image_path: str) -> dict:
        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        batch = self._prepare(image_path)
        preprocessing_ms = (
            time.perf_counter() - preprocess_start
        ) * 1000

        inference_start = time.perf_counter()
        if self.config.model_type == "keras":
            output = self._keras_predict(batch)
        else:
            output = self._tflite_predict(batch)
        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000

        probabilities = self._probabilities(output)

        if len(probabilities) != len(self.config.class_names):
            raise ValueError(
                "Model output count and class_names count differ"
            )

        predicted_index = int(np.argmax(probabilities))
        total_ms = (time.perf_counter() - total_start) * 1000

        return {
            "model": self.config.name,
            "model_type": self.config.model_type,
            "image_path": str(
                Path(image_path).expanduser().resolve()
            ),
            "predicted_class": self.config.class_names[predicted_index],
            "confidence": round(
                float(probabilities[predicted_index]), 6
            ),
            "probabilities": {
                name: round(float(probability), 6)
                for name, probability in zip(
                    self.config.class_names,
                    probabilities,
                )
            },
            "timing_ms": {
                "preprocessing": round(preprocessing_ms, 3),
                "inference": round(inference_ms, 3),
                "total": round(total_ms, 3),
            },
        }
