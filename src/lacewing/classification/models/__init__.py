"""Model registry. Map model name -> (factory, expected feature kind)."""
from __future__ import annotations

from torch import nn

from . import (ann, autoencoder, cnn1d, cnn2d_spectrogram, fcn,
               inception_time, resnet,
               transformer, transformer_patch,
               cnn_transformer_parallel, cnn_transformer_sequential,
               gru, cnn_gru_parallel, cnn_gru_sequential)


# (factory(input_len_or_shape) -> nn.Module, features required)
REGISTRY: dict[str, tuple] = {
    "ann":                 (ann.build,                        "raw"),
    "cnn1d":               (cnn1d.build,                      "raw"),
    "fcn":                 (fcn.build,                        "raw"),
    "resnet":              (resnet.build,                     "raw"),
    "inception":           (inception_time.build,             "raw"),
    "autoencoder":         (autoencoder.build,                "raw"),
    "cnn2d_spec":          (cnn2d_spectrogram.build,          "spectrogram"),
    "transformer":         (transformer.build,                "raw"),
    "transformer_patch":   (transformer_patch.build,          "raw"),
    "cnn_transformer_par": (cnn_transformer_parallel.build,   "raw"),
    "cnn_transformer_seq": (cnn_transformer_sequential.build, "raw"),
    "gru":                 (gru.build,                        "raw"),
    "cnn_gru_par":         (cnn_gru_parallel.build,           "raw"),
    "cnn_gru_seq":         (cnn_gru_sequential.build,         "raw"),
}


def build(model_name: str, input_shape) -> nn.Module:
    if model_name not in REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(REGISTRY)}.")
    factory, _ = REGISTRY[model_name]
    return factory(input_shape)


def features_for(model_name: str) -> str:
    return REGISTRY[model_name][1]


def list_models() -> list[str]:
    return list(REGISTRY)
