"""Builds a serving directory with the loop-1 veto on top of the exit-lab one (boptim_lcb/make_exitlab_modeling.py; that
directory is not touched). Same released weights (symlink). One addition to the decode-time early exit, off by default:

  decode_exit_veto     = path.pt    harm probe of randrouter/harm_fit.py ({"kind", "state_dict", "mean", "std"})
  decode_exit_veto_tau = float      a token the gate would release after loop 1 stays for loop 2 when the probe's score of its
                                    loop-1 state is above tau (the probe predicts what the exit costs the later tokens)

  python randrouter/make_exitlab_veto.py [DST]           (default /home/jli199/boptim_scratch/ouro_base_exitlab_veto)
"""
import os
import py_compile
import shutil
import sys

SRC = "/home/jli199/boptim_scratch/ouro_base_exitlab3"
DST = sys.argv[1] if len(sys.argv) > 1 else "/home/jli199/boptim_scratch/ouro_base_exitlab_veto"

s = open(f"{SRC}/modeling_ouro.py", encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new, 1)


# 1. the veto, after the gate / confidence decision
rep("""                        self._dee_conf_checks = getattr(self, "_dee_conf_checks", 0) + 1
                if exit_now:
""",
    """                        self._dee_conf_checks = getattr(self, "_dee_conf_checks", 0) + 1
                # loop-1 veto (randrouter/make_exitlab_veto.py): the gate judges the token's own readout, the probe what its
                # stand-in keys/values would cost the later tokens
                if exit_now and current_ut == 0 and getattr(self, "decode_exit_veto", None):
                    if self._exitlab_veto_score(hidden_states[0, -1]) > float(getattr(self, "decode_exit_veto_tau", 0.0)):
                        exit_now = False
                        self._dee_vetoed = getattr(self, "_dee_vetoed", 0) + 1
                if exit_now:
""")

# 2. settings and the probe loader
rep("""        self._fill_cache = None
        self._stream_cache = None
        # Initialize weights and apply final processing
        self.post_init()
""",
    """        self._fill_cache = None
        self._stream_cache = None
        self.decode_exit_veto = getattr(self.config, "decode_exit_veto", None)
        self.decode_exit_veto_tau = getattr(self.config, "decode_exit_veto_tau", 0.0)
        self._veto_cache = None
        # Initialize weights and apply final processing
        self.post_init()

    def _exitlab_veto_score(self, h):
        \"\"\"Score of the harm probe for one normed loop-1 state [dim] (float).\"\"\"
        path = self.decode_exit_veto
        if self._veto_cache is None or self._veto_cache[0] != path:
            blob = torch.load(path, map_location="cpu")
            sd = blob["state_dict"]
            dev = self.early_exit_gate.weight.device
            if blob["kind"] == "mlp":
                net = nn.Sequential(nn.Linear(sd["0.weight"].shape[1], sd["0.weight"].shape[0]), nn.GELU(), nn.Linear(sd["2.weight"].shape[1], 1))
            else:
                net = nn.Linear(sd["weight"].shape[1], 1)
            net.load_state_dict(sd)
            self._veto_cache = (path, net.to(dev).float().eval(), blob["mean"].to(dev).float(), blob["std"].to(dev).float())
        _, net, mean, std = self._veto_cache
        return float(net((h.float()[None] - mean) / std).squeeze())
""")

os.makedirs(DST, exist_ok=True)
for fn in os.listdir(SRC):
    if fn in ("modeling_ouro.py", "__pycache__"):
        continue
    src, dst = os.path.join(SRC, fn), os.path.join(DST, fn)
    if os.path.lexists(dst):
        os.remove(dst)
    if os.path.islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        shutil.copy(src, dst)
tmp = os.path.join(DST, "modeling_ouro.py.new")
open(tmp, "w", encoding="utf-8").write(s)
py_compile.compile(tmp, doraise=True)
os.replace(tmp, os.path.join(DST, "modeling_ouro.py"))
print(f"wrote {DST}/modeling_ouro.py and {sorted(os.listdir(DST))}")
