# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import Gemma3nTextArgs


class Gemma3nStateDictAdapter(StateDictAdapter):
    """State dict adapter for Gemma 3N text-only model.

    This adapter converts between TorchTitan's native key format and the
    Hugging Face Gemma 3N checkpoint format. It focuses on the text portion
    ("language_model") and ignores vision/audio parts if present in the HF
    checkpoint.

    Key principles:
    - HF prefixes text model parameters with "model.language_model.".
    - TorchTitan's native keys do not have this prefix.
    - The token embedding is named "embed_tokens" in HF and "tok_embeddings"
      in TorchTitan. Most other submodule names align 1:1.
    - HF Gemma 3N checkpoints do not include a separate lm_head parameter; the
      logits projection is typically tied to embeddings. TorchTitan exposes an
      "output" Linear whose weight is tied to embeddings when configured to do
      so. We therefore drop/ignore any output head in conversions.
    - No q/k permutation (as used for some Llama variants) is required.
    """

    HF_TEXT_PREFIX = "model.language_model."

    def __init__(self, model_args: Gemma3nTextArgs):
        self.model_args = model_args

    # ----------------------------
    # Helper key mapping functions
    # ----------------------------
    @classmethod
    def _hf_to_native_key(cls, hf_key: str) -> str | None:
        """Map a single HF key to native TorchTitan key.

        Returns None if the key is not part of the Gemma 3N text model.
        """
        if not hf_key.startswith(cls.HF_TEXT_PREFIX):
            return None

        subkey = hf_key[len(cls.HF_TEXT_PREFIX) :]

        # Embedding rename: embed_tokens.weight -> tok_embeddings.weight
        if subkey == "embed_tokens.weight":
            return "tok_embeddings.weight"

        # Everything else under language_model maps 1:1 to native names
        # Examples:
        #  - layers.{L}.self_attn.q_proj.weight -> layers.{L}.self_attn.q_proj.weight
        #  - embed_tokens_per_layer.weight -> embed_tokens_per_layer.weight
        #  - norm.weight -> norm.weight
        #  - per_layer_model_projection.weight -> per_layer_model_projection.weight
        #  - per_layer_projection_norm.weight -> per_layer_projection_norm.weight
        #  - altup_projections.{i}.weight -> altup_projections.{i}.weight
        #  - altup_unembed_projections.{i}.weight -> altup_unembed_projections.{i}.weight
        return subkey

    @classmethod
    def _native_to_hf_key(cls, native_key: str) -> str | None:
        """Map a single native TorchTitan key to an HF key.

        Returns None if the key should be omitted in the HF checkpoint.
        """
        # Drop non-text or internal buffers if they appear
        if native_key.startswith("rotary_global.") or native_key.startswith("rotary_local."):
            return None

        # TorchTitan exposes an output head which is tied to embeddings.
        # HF Gemma 3N does not store a separate head in the checkpoints we target.
        if native_key == "output.weight":
            return None

        # Embedding rename: tok_embeddings.weight -> embed_tokens.weight
        if native_key == "tok_embeddings.weight":
            return f"{cls.HF_TEXT_PREFIX}embed_tokens.weight"

        # All other keys are prefixed with model.language_model.
        return f"{cls.HF_TEXT_PREFIX}{native_key}"

    # --------------
    # Main API
    # --------------
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert native TorchTitan state dict to Hugging Face format."""
        hf_state_dict: dict[str, Any] = {}

        for key, value in state_dict.items():
            hf_key = self._native_to_hf_key(key)
            if hf_key is None:
                continue
            hf_state_dict[hf_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert Hugging Face state dict to native TorchTitan format.

        This only consumes the Gemma 3N text portion: keys under
        "model.language_model.". Keys for other modalities (vision, audio) are
        ignored.
        """
        state_dict: dict[str, Any] = {}

        # Pre-compile regex used in some sanity checks (not strictly required)
        layer_key_regex = re.compile(r"^layers\.\d+\.")

        for hf_key, value in hf_state_dict.items():
            native_key = self._hf_to_native_key(hf_key)
            if native_key is None:
                continue

            # Optional sanity: if it's a layer key, ensure the pattern matches
            # "layers.{L}.<submodule>". We do not remap submodule names because
            # our implementation mirrors HF naming for Gemma 3N.
            if layer_key_regex.match(native_key):
                state_dict[native_key] = value
                continue

            # Top-level text keys (embeddings, norms, projections, altup):
            state_dict[native_key] = value

        return state_dict

