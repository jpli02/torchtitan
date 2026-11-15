"""Builds an experimental serving directory, ouro_base_exitlab, from the deployed early-exit one (ouro_base_kvshare; the deployed
file is not touched). Same released weights (symlink), two additions to the decode-time early exit, both off by default:

  decode_exit_conf  = [t1, t2, t3]   a token also leaves after loop s when the language-model head's top-1 probability after
                                     that loop is at least t_s (1.0 = off for that loop). The released gate's cumulative exit
                                     probability saturates near s/4 and cannot tell tokens apart; the head's own confidence can.
  decode_fill_maps  = path.pt        "calibrated fill": the cache slots of the loops a token skips are filled with affine maps
                                     of its exit-loop keys and values (fitted by kv_fill_fit.py on full-depth answers) instead
                                     of plain copies. Keys are mapped in the unrotated space (rotary rotation undone and redone).

Gate thresholds keep working as before; to route on confidence alone give the gate thresholds [0.999, 0.999, 0.999, 1.0].
  python boptim_lcb/make_exitlab_modeling.py            (idempotent; compiles the result before it is put in place)
"""
import os
import py_compile
import shutil

SRC = "/home/jli199/boptim_scratch/ouro_base_kvshare"
DST = "/home/jli199/boptim_scratch/ouro_base_exitlab"

s = open(f"{SRC}/modeling_ouro.py", encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new, 1)


# 1. exit decision: gate threshold OR head confidence
rep("""                if bool(dee_cumulative >= (dee_list[current_ut] if dee_list else float(dee))):
                    if isinstance(past_key_values, SharedKVCache):""",
    """                exit_now = bool(dee_cumulative >= (dee_list[current_ut] if dee_list else float(dee)))
                # exit lab: leave also when the language-model head is confident enough after this loop
                conf_thr = getattr(self, "decode_exit_conf", None)
                if not exit_now and conf_thr is not None and getattr(self, "_dee_head", None) is not None:
                    tau = float(conf_thr[current_ut]) if isinstance(conf_thr, (list, tuple)) else float(conf_thr)
                    if tau < 1.0:
                        pr = torch.softmax(self._dee_head(hidden_states[0, -1]).float(), dim=-1)
                        exit_now = bool(pr.max() >= tau)
                        self._dee_conf_checks = getattr(self, "_dee_conf_checks", 0) + 1
                if exit_now:
                    if isinstance(past_key_values, SharedKVCache):""")

# 2. cache fill of the skipped loops: calibrated maps when given, plain copies otherwise
rep("""                                kc[dst] = torch.cat([kc[dst], kc[src][:, :, -1:, :]], dim=2)
                                vc[dst] = torch.cat([vc[dst], vc[src][:, :, -1:, :]], dim=2)""",
    """                                k_new, v_new = kc[src][:, :, -1:, :], vc[src][:, :, -1:, :]
                                fm = self._exitlab_maps()
                                if fm is not None and (current_ut, s) in fm[0]:
                                    k_new, v_new = self._exitlab_fill(fm, current_ut, s, l, k_new, v_new, position_embeddings)
                                kc[dst] = torch.cat([kc[dst], k_new], dim=2)
                                vc[dst] = torch.cat([vc[dst], v_new], dim=2)""")

# 3. settings and helpers on the inner model
rep("""        self.kv_collapse_prefill = bool(getattr(self.config, "kv_collapse_prefill", False))
        # Initialize weights and apply final processing
        self.post_init()
""",
    """        self.kv_collapse_prefill = bool(getattr(self.config, "kv_collapse_prefill", False))
        # exit lab (boptim_lcb/make_exitlab_modeling.py): confidence exits and calibrated cache fill, both off by default
        self.decode_exit_conf = getattr(self.config, "decode_exit_conf", None)
        self.decode_fill_maps = getattr(self.config, "decode_fill_maps", None)
        self._fill_cache = None
        # Initialize weights and apply final processing
        self.post_init()

    def _exitlab_maps(self):
        \"\"\"(key maps, value maps) of the calibrated fill, {(exit loop, target loop): [layers, heads, dim + 1, dim]}, or None.\"\"\"
        path = getattr(self, "decode_fill_maps", None)
        if not path:
            return None
        if self._fill_cache is None or self._fill_cache[0] != path:
            blob = torch.load(path, map_location="cpu")
            dev = self.early_exit_gate.weight.device
            km, vm = {}, {}
            for name, m in blob["maps"].items():
                w, a, b = (int(x) for x in name.split(":"))
                (km if w == 0 else vm)[(a, b)] = m.to(dev, torch.float32)
            self._fill_cache = (path, km, vm)
        return self._fill_cache[1], self._fill_cache[2]

    @staticmethod
    def _exitlab_fill(fm, e, s, l, k_new, v_new, position_embeddings):
        \"\"\"Stand-ins for the loop-s key and value of a token that left after loop e: affine maps of its loop-e key and value.
        Cached keys carry the rotary rotation of their position and the maps were fitted on unrotated keys, so the key is
        rotated back, mapped and rotated again.\"\"\"
        km, vm = fm
        cos, sin = position_embeddings
        cos, sin = cos[:, None, -1:, :].float(), sin[:, None, -1:, :].float()          # [1, 1, 1, dim]
        dt = k_new.dtype
        k = k_new.float()
        half = k.shape[-1] // 2
        k = k * cos - torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin
        A = km[(e, s)][l]                                                               # [heads, dim + 1, dim]
        k = torch.einsum("bhtd,hde->bhte", k, A[:, :-1, :]) + A[:, -1, :][None, :, None, :]
        k = k * cos + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin
        B = vm[(e, s)][l]
        v = torch.einsum("bhtd,hde->bhte", v_new.float(), B[:, :-1, :]) + B[:, -1, :][None, :, None, :]
        return k.to(dt), v.to(dt)
""")

# 4. the wrapper hands its head to the inner model (not registered as a sub-module there: the weights stay where they are)
rep("""        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
""",
    """        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        object.__setattr__(self.model, "_dee_head", self.lm_head)
""")

os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    if f in ("modeling_ouro.py", "__pycache__") or f.startswith("modeling_ouro.py."):
        continue
    src, dst = f"{SRC}/{f}", f"{DST}/{f}"
    if os.path.lexists(dst):
        continue
    if os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        shutil.copy(src, dst)
tmp = f"{DST}/modeling_ouro.py.new"
open(tmp, "w", encoding="utf-8").write(s)
py_compile.compile(tmp, doraise=True)
os.replace(tmp, f"{DST}/modeling_ouro.py")
print("wrote", f"{DST}/modeling_ouro.py", "and", sorted(os.listdir(DST)))
