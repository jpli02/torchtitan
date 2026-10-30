"""Build serving dirs whose modeling_ouro.py adds decode-time KV sharing (Ouro paper
Table 14, 'last-step reuse') on top of the true-early-exit patch.

  python make_kvshare_modeling.py
    -> /home/jli199/boptim_scratch/ouro_v3_kvshare    (weights -> ouro_tb80sft26b_v3_clean)
    -> /home/jli199/boptim_scratch/ouro_base_kvshare  (weights -> assets/hf/Ouro-2.6B-Thinking)

New config flags (default off = stock behaviour):
  kv_share_decode    : decoded tokens keep only their final/exit-step K/V, shared by all
                       recurrence steps (4x less cache for generated tokens)
  kv_collapse_prefill: after a prompt chunk is prefilled with per-step K/V, keep only its
                       last-step K/V for later attention (4x less cache for prompt tokens;
                       accuracy impact to be measured)
"""
import os
import shutil
import sys

SRC = "/home/jli199/boptim_scratch/ouro_v3_earlyexit/modeling_ouro.py"
BASE = "/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking"
V3 = "/home/jli199/boptim_scratch/ouro_tb80sft26b_v3_clean"
OUT = {"/home/jli199/boptim_scratch/ouro_v3_kvshare": V3, "/home/jli199/boptim_scratch/ouro_base_kvshare": BASE}

src = open(SRC, encoding="utf-8").read()

CACHE_CLASS = '''
class SharedKVCache(Cache):
    """Segment cache for Ouro with decode-time KV sharing (Ouro paper Sec. 5.4.2).

    Per layer, the sequence is a list of segments in order:
      per-step segment : a prefilled prompt chunk, K/V kept for every recurrence step
      shared segment   : decoded tokens, one K/V (their final or exit step) used by all steps
    With collapse_prefill, a prompt chunk becomes a shared segment (last-step K/V) once all
    its steps have been computed. Slot index = step * n_layers + layer, as in the stock cache.
    """

    def __init__(self, n_layers: int, total_ut_steps: int, share_decode: bool = True,
                 collapse_prefill: bool = False):
        self.n_layers = int(n_layers)
        self.T = int(total_ut_steps)
        self.share_decode = bool(share_decode)
        self.collapse_prefill = bool(collapse_prefill)
        self.segs: list[list[dict]] = [[] for _ in range(self.n_layers)]
        self.pending: dict = {}
        self.layers: list = []
        self.key_cache: list = []
        self.value_cache: list = []
        self._seen_tokens = 0

    @staticmethod
    def _k(seg, s):
        return seg["k"] if seg["shared"] else seg["k"][s]

    @staticmethod
    def _v(seg, s):
        return seg["v"] if seg["shared"] else seg["v"][s]

    def _seg_len(self, seg):
        return seg["k"].shape[2] if seg["shared"] else seg["k"][0].shape[2]

    def _commit_layer(self, l, k, v):
        segs = self.segs[l]
        if segs and segs[-1]["shared"]:
            segs[-1]["k"] = torch.cat([segs[-1]["k"], k], dim=2)
            segs[-1]["v"] = torch.cat([segs[-1]["v"], v], dim=2)
        else:
            segs.append({"shared": True, "k": k, "v": v})

    def commit_decode(self, step: int) -> None:
        """Commit the current decode token's K/V from `step` (exit step) for every layer."""
        for l in range(self.n_layers):
            kv = self.pending.pop((step, l), None)
            if kv is not None:
                self._commit_layer(l, kv[0], kv[1])
        self.pending.clear()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        s, l = divmod(int(layer_idx), self.n_layers)
        n = key_states.shape[2]
        segs = self.segs[l]
        if n > 1:
            if s == 0:
                segs.append({"shared": False, "k": [None] * self.T, "v": [None] * self.T})
            seg = segs[-1]
            seg["k"][s] = key_states
            seg["v"][s] = value_states
            past = segs[:-1]
            k_full = torch.cat([self._k(x, s) for x in past] + [key_states], dim=2) if past else key_states
            v_full = torch.cat([self._v(x, s) for x in past] + [value_states], dim=2) if past else value_states
            if s == self.T - 1 and self.collapse_prefill:
                seg["shared"], seg["k"], seg["v"] = True, key_states, value_states
            self._seen_tokens = k_full.shape[2]
            return k_full, v_full
        # decode token (n == 1): hold per-step K/V until the final/exit step commits it
        if s == 0 and l == 0:
            self.pending.clear()
        self.pending[(s, l)] = (key_states, value_states)
        k_full = torch.cat([self._k(x, s) for x in segs] + [key_states], dim=2) if segs else key_states
        v_full = torch.cat([self._v(x, s) for x in segs] + [value_states], dim=2) if segs else value_states
        if s == self.T - 1:
            self.pending.pop((s, l), None)
            self._commit_layer(l, key_states, value_states)
        self._seen_tokens = k_full.shape[2]
        return k_full, v_full

    def get_seq_length(self, layer_idx=0):
        l = 0 if layer_idx is None else int(layer_idx) % self.n_layers
        return sum(self._seg_len(seg) for seg in self.segs[l])

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    def get_max_length(self):
        return None

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx):
        for segs in self.segs:
            for seg in segs:
                if seg["shared"]:
                    seg["k"] = seg["k"].index_select(0, beam_idx.to(seg["k"].device))
                    seg["v"] = seg["v"].index_select(0, beam_idx.to(seg["v"].device))
                else:
                    seg["k"] = [x.index_select(0, beam_idx.to(x.device)) if x is not None else None for x in seg["k"]]
                    seg["v"] = [x.index_select(0, beam_idx.to(x.device)) if x is not None else None for x in seg["v"]]

    @property
    def is_compileable(self) -> bool:
        return False

    def cache_bytes(self) -> int:
        tot = 0
        for segs in self.segs:
            for seg in segs:
                ts = [seg["k"], seg["v"]] if seg["shared"] else [x for x in seg["k"] + seg["v"] if x is not None]
                tot += sum(t.numel() * t.element_size() for t in ts)
        return tot

    def clear(self) -> None:
        self.segs = [[] for _ in range(self.n_layers)]
        self.pending.clear()
        self._seen_tokens = 0


'''


def rep(old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        sys.exit(f"anchor x{n}, expected {count}: {old[:70]!r}")
    src = src.replace(old, new)


rep("\ndef eager_attention_forward(", CACHE_CLASS + "\ndef eager_attention_forward(")
rep("    if isinstance(cache, UniversalTransformerCache):\n        return False\n",
    "    if isinstance(cache, (UniversalTransformerCache, SharedKVCache)):\n        return False\n")
rep('        self.decode_early_exit_threshold = getattr(\n            self.config, "decode_early_exit_threshold", None)\n',
    '        self.decode_early_exit_threshold = getattr(\n            self.config, "decode_early_exit_threshold", None)\n'
    '        self.kv_share_decode = bool(getattr(self.config, "kv_share_decode", False))\n'
    '        self.kv_collapse_prefill = bool(getattr(self.config, "kv_collapse_prefill", False))\n')
rep("            if needs_universal_cache(past_key_values, max_cache_size):\n"
    "                past_key_values = UniversalTransformerCache(max_cache_size)\n",
    "            if getattr(self, \"kv_share_decode\", False) or getattr(self, \"kv_collapse_prefill\", False):\n"
    "                if not isinstance(past_key_values, SharedKVCache):\n"
    "                    if past_key_values is not None and past_key_values.get_seq_length() > 0:\n"
    "                        raise ValueError(\"KV sharing needs a SharedKVCache from the first forward\")\n"
    "                    past_key_values = SharedKVCache(total_layers, total_ut_steps, share_decode=True,\n"
    "                                                    collapse_prefill=getattr(self, \"kv_collapse_prefill\", False))\n"
    "            elif needs_universal_cache(past_key_values, max_cache_size):\n"
    "                past_key_values = UniversalTransformerCache(max_cache_size)\n")
rep("            _kc = past_key_values.key_cache\n"
    "            _n = self.total_ut_steps * n_layers\n"
    "            do_dee = len(_kc) >= _n and all(x is not None for x in _kc[:_n])\n",
    "            if isinstance(past_key_values, SharedKVCache):\n"
    "                do_dee = past_key_values.get_seq_length() > 0\n"
    "            else:\n"
    "                _kc = past_key_values.key_cache\n"
    "                _n = self.total_ut_steps * n_layers\n"
    "                do_dee = len(_kc) >= _n and all(x is not None for x in _kc[:_n])\n")
rep("                if bool(dee_cumulative >= float(dee)):\n"
    "                    kc = past_key_values.key_cache\n"
    "                    vc = past_key_values.value_cache\n"
    "                    for s in range(current_ut + 1, self.total_ut_steps):\n"
    "                        for l in range(n_layers):\n"
    "                            src = current_ut * n_layers + l\n"
    "                            dst = s * n_layers + l\n"
    "                            kc[dst] = torch.cat([kc[dst], kc[src][:, :, -1:, :]], dim=2)\n"
    "                            vc[dst] = torch.cat([vc[dst], vc[src][:, :, -1:, :]], dim=2)\n",
    "                if bool(dee_cumulative >= float(dee)):\n"
    "                    if isinstance(past_key_values, SharedKVCache):\n"
    "                        past_key_values.commit_decode(current_ut)\n"
    "                    else:\n"
    "                        kc = past_key_values.key_cache\n"
    "                        vc = past_key_values.value_cache\n"
    "                        for s in range(current_ut + 1, self.total_ut_steps):\n"
    "                            for l in range(n_layers):\n"
    "                                src = current_ut * n_layers + l\n"
    "                                dst = s * n_layers + l\n"
    "                                kc[dst] = torch.cat([kc[dst], kc[src][:, :, -1:, :]], dim=2)\n"
    "                                vc[dst] = torch.cat([vc[dst], vc[src][:, :, -1:, :]], dim=2)\n")

for out, weights in OUT.items():
    os.makedirs(out, exist_ok=True)
    open(f"{out}/modeling_ouro.py", "w", encoding="utf-8").write(src)
    for f in ("config.json", "configuration_ouro.py", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors.index.json"):
        p = f"{weights}/{f}"
        if os.path.exists(p):
            shutil.copy(p, f"{out}/{f}")
    for f in sorted(os.listdir(weights)):
        if f.endswith(".safetensors"):
            dst = f"{out}/{f}"
            if os.path.lexists(dst):
                os.unlink(dst)
            os.symlink(f"{weights}/{f}", dst)
    print(f"wrote {out}: modeling {len(src.splitlines())} lines, weights -> {weights}")
