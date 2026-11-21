"""Serving directory whose exit gate can be any router architecture: /home/jli199/boptim_scratch/ouro_base_gatelab.

The released Ouro-2.6B serving model (ouro_base_kvshare: released weights, true decode-time early exit, exact loop counter) builds its
exit gate as nn.Linear(2048, 1). For the gate search (objective ouro_lcb_gate_nas: Stage-II training of a router of any Tier-1
architecture, model frozen) the decoder must run a trained router instead. This writes a copy of the serving directory (weights and
tokenizer files linked, not copied) whose modeling file adds ONE option:

    decode_router=<file.pt>   {"spec": {layers, hidden, act, norm}, "state_dict": ...} saved by randrouter/nas_fit.save_router;
                              on the first forward the router is built with the same construction as
                              torchtitan/models/ouro/model.py::build_router, loaded, cast to the model's dtype and device, and
                              replaces model.early_exit_gate (input: the normed state after each loop, exactly what the released
                              gate reads). Unset = the released gate, byte-identical behaviour.

  /home/jli199/boptim-agent/.venv/bin/python boptim_lcb/make_gatelab_modeling.py
"""
import os
import shutil

SRC = "/home/jli199/boptim_scratch/ouro_base_kvshare"
DST = "/home/jli199/boptim_scratch/ouro_base_gatelab"

HELPER = '''

def _gatelab_build_router(dim, spec):
    """Router construction of torchtitan/models/ouro/model.py::build_router (Tier-1 NAS axes)."""
    layers = int(spec.get("layers", 0))
    hidden = int(spec.get("hidden", 128))
    act = str(spec.get("act", "gelu")).lower()
    norm = bool(spec.get("norm", False))
    if layers <= 0 and not norm:
        return nn.Linear(dim, 1, bias=True)
    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
    mods = []
    if norm:
        mods.append(nn.LayerNorm(dim))
    d = dim
    for _ in range(layers):
        mods.append(nn.Linear(d, hidden))
        mods.append(acts[act]())
        d = hidden
    mods.append(nn.Linear(d, 1))
    return nn.Sequential(*mods)

'''

HOOK = '''        # gate lab: a trained router of any Tier-1 architecture replaces the released gate (decode_router=<file.pt>)
        _rp = getattr(self.config, "decode_router", None)
        if _rp and getattr(self, "_gatelab_router_path", None) != _rp:
            _d = torch.load(_rp, map_location="cpu")
            _m = _gatelab_build_router(self.config.hidden_size, _d["spec"])
            _m.load_state_dict(_d["state_dict"])
            _w = self.embed_tokens.weight
            self.early_exit_gate = _m.to(device=_w.device, dtype=_w.dtype).eval()
            self._gatelab_router_path = _rp
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
'''

ANCHOR = '''        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
'''


def main():
    os.makedirs(DST, exist_ok=True)
    for name in os.listdir(SRC):
        if name in ("modeling_ouro.py", "__pycache__") or name.endswith(".before_exec_counter"):
            continue
        dst = os.path.join(DST, name)
        if os.path.lexists(dst):
            os.remove(dst)
        src = os.path.realpath(os.path.join(SRC, name))
        if name.endswith(".safetensors"):
            os.symlink(src, dst)
        else:
            shutil.copy(src, dst)
    s = open(os.path.join(SRC, "modeling_ouro.py")).read()
    assert s.count(ANCHOR) == 1, "anchor not found once in the serving modeling file"
    anchor_cls = "\n@auto_docstring\nclass OuroModel(OuroPreTrainedModel):"     # the helper goes above the class's decorator
    if s.count(anchor_cls) != 1:
        anchor_cls = "\nclass OuroModel(OuroPreTrainedModel):"
    assert s.count(anchor_cls) == 1, "OuroModel class not found once"
    s = s.replace(anchor_cls, HELPER + anchor_cls, 1)
    s = s.replace(ANCHOR, HOOK, 1)
    with open(os.path.join(DST, "modeling_ouro.py"), "w") as f:
        f.write(s)
    import py_compile
    py_compile.compile(os.path.join(DST, "modeling_ouro.py"), doraise=True)
    print("wrote", DST, sorted(os.listdir(DST)))


if __name__ == "__main__":
    main()
