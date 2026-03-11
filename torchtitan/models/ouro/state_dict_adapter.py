import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

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

        return state_dict
