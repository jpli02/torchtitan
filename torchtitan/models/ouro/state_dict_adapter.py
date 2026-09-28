import json
import os
import re
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from torchtitan.tools.logging import logger

from .model import OuroModel


class OuroStateDictAdapter(StateDictAdapter):
    """Adapter for converting between Ouro HuggingFace and TorchTitan formats."""

    def __init__(
        self,
        model_config: OuroModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.input_layernorm.weight",
            "model.layers.{}.input_layernorm_2.weight": "layers.{}.input_layernorm_2.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.weight",
            "model.layers.{}.post_attention_layernorm_2.weight": "layers.{}.post_attention_layernorm_2.weight",
            "model.norm.weight": "norm.weight",
            "model.early_exit_gate.weight": "early_exit_gate.weight",
            "model.early_exit_gate.bias": "early_exit_gate.bias",
            "lm_head.weight": "output.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                if abstract_key not in to_hf_map:
                    continue
                new_key = to_hf_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value
            else:
                if key not in to_hf_map:
                    continue
                new_key = to_hf_map[key]
                if new_key is None:
                    continue
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in self.from_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value
            else:
                if key not in self.from_hf_map:
                    continue
                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue
                state_dict[new_key] = value

        # Router-architecture NAS: when OURO_ROUTER_SPEC decodes to something
        # other than the original nn.Linear(dim,1) gate, the pretrained weights
        # can't be loaded into the new module shape -- drop early_exit_gate.*
        # so the newly-built router keeps its random init and SFT trains it
        # from scratch. (Backbone still loads pretrained either way.)
        #
        # BUG FIXED: this used to trigger on OURO_ROUTER_SPEC being *set at
        # all*, not on the spec actually differing from the default. A NAS
        # candidate that decodes to layers=0 (shape-identical to the original
        # linear gate) was silently discarding the pretrained gate and training
        # from scratch anyway -- artificially handicapping every "linear"
        # architecture point in a NAS search relative to what it should score.
        _raw_spec = os.environ.get("OURO_ROUTER_SPEC")
        _is_default_gate = True
        if _raw_spec:
            try:
                _spec = json.loads(_raw_spec)
                _is_default_gate = int(_spec.get("layers", 0)) <= 0 and not _spec.get("norm", False)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"OURO_ROUTER_SPEC={_raw_spec!r} not valid JSON ({e}); "
                                "treating as default gate shape.")
        if _raw_spec and not _is_default_gate:
            dropped = [k for k in state_dict if k.startswith("early_exit_gate")]
            for k in dropped:
                del state_dict[k]
            if dropped:
                logger.info(
                    f"OURO_ROUTER_SPEC={_raw_spec}: dropped pretrained gate keys "
                    f"{dropped} (custom router random-init, trained from scratch)."
                )
        # Optionally start SFT from a RANDOMLY-INITIALISED exit-gate/router instead
        # of the pretrained one: overwrite the loaded early_exit_gate.{weight,bias}
        # with a fresh nn.Linear(dim, 1) init (backbone still loads pretrained).
        # Keys are kept (not dropped) so the checkpoint load stays non-strict-safe.
        elif os.environ.get("OURO_RANDOM_INIT_GATE", "0") == "1":
            wk, bk = "early_exit_gate.weight", "early_exit_gate.bias"
            if wk in state_dict:
                w = state_dict[wk]
                dim = w.shape[-1]
                gate = torch.nn.Linear(dim, 1, bias=(bk in state_dict))
                state_dict[wk] = gate.weight.detach().to(dtype=w.dtype)
                if bk in state_dict:
                    state_dict[bk] = gate.bias.detach().to(dtype=state_dict[bk].dtype)
                logger.info(
                    "OURO_RANDOM_INIT_GATE=1: early_exit_gate re-initialised randomly "
                    "(router trained from scratch during SFT; backbone still pretrained)."
                )

        return state_dict
