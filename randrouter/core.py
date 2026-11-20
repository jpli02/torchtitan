"""Random-router training of the looped model: the shared machinery.

WHY. A router search over the released gate's thresholds cannot beat the released gate on LiveCodeBench (2026-11): rules that
win when every earlier token still has its exact keys and values ("gather" mode) lose in real decoding, where a token that left
after loop s has, in the cache slots of the loops it skipped, a copy of its loop-s keys and values. The released model was
trained with every token at full depth, so it never saw such a cache. Here the model is trained WITH that cache: every answer
token gets an exit loop from a router (a random one, by default), the cache is filled exactly as the decoder fills it, and the
loss asks every loop the token did run to predict the full-depth answer.

WHAT IS EMULATED. Decoding (boptim_scratch/ouro_base_kvshare/modeling_ouro.py, decode_early_exit_threshold):
  * the prompt (prefill) always runs the 4 loops;
  * a decoded token runs loops 0..e and leaves; for every skipped loop s > e and every layer l its cache slot (s, l) gets a copy
    of its keys and values of slot (e, l); its next-token prediction is read from the loop-e state.
The teacher-forced parallel forward below reproduces this for a whole sequence at once: at loop t every layer computes fresh
keys and values for all positions and then replaces those of the positions that already left by the ones carried from the
previous loop. A position's state at loop t depends only on EARLIER positions' keys and values at loop t, so the states of all
loops a position did run are exactly what the decoder computes (the state of a position at a loop it did not run is garbage and
is masked everywhere). With thresholds instead of a plan the exits are decided loop by loop from the gate, as in the decoder.

Sequences are right-padded in a batch; with causal attention no real position sees the padding.
"""
from __future__ import annotations

import json
import math
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

SERVE_DIR = "/home/jli199/boptim_scratch/ouro_base_kvshare"            # released weights (symlink) + the serving model file
BASE_DIR = "/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking"      # the released base model (every training starts here)
N_LOOPS = 4


def load_model(ckpt: str = SERVE_DIR, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    return model, tok


def _rotary(model):
    return sys.modules[type(model).__module__].apply_rotary_pos_emb


def load_fill_maps(path: str, device: str = "cuda"):
    """Maps of the calibrated fill as ({(exit loop, target loop): [layers, heads, dim + 1, dim]} for keys, the same for values)."""
    blob = torch.load(path, map_location="cpu")
    km, vm = {}, {}
    for name, m in blob["maps"].items():
        w, a, b = (int(x) for x in name.split(":"))
        (km if w == 0 else vm)[(a, b)] = m.to(device, torch.float32)
    return km, vm


def _rot_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _mapped_fill(fill, k_carry, v_carry, cos, sin):
    """Affine stand-ins for this loop's keys/values of the positions that left earlier, from their exit loop's (carried) ones."""
    exit_, t, l, km, vm = fill
    c, s_ = cos[:, None].float(), sin[:, None].float()
    k_un = k_carry.float() * c - _rot_half(k_carry.float()) * s_          # undo the rotary rotation
    v32 = v_carry.float()
    k_out, v_out = k_carry, v_carry
    for e in range(t):
        m = (exit_ == e)[:, None, :, None]
        if not bool(m.any()):
            continue
        a_k, a_v = km[(e, t)][l], vm[(e, t)][l]                            # [heads, dim + 1, dim]
        k_m = torch.einsum("bhtd,hde->bhte", k_un, a_k[:, :-1, :]) + a_k[:, -1, :][None, :, None, :]
        k_m = k_m * c + _rot_half(k_m) * s_
        v_m = torch.einsum("bhtd,hde->bhte", v32, a_v[:, :-1, :]) + a_v[:, -1, :][None, :, None, :]
        k_out = torch.where(m, k_m.to(k_carry.dtype), k_out)
        v_out = torch.where(m, v_m.to(v_carry.dtype), v_out)
    return k_out, v_out


def _block(layer, rot, h, cos, sin, k_prev, v_prev, stale, fill=None):
    """One decoder layer on the whole sequence; keys/values of the positions in `stale` are replaced by the carried ones
    (plain copy of their exit loop's) or, with `fill`, by the calibrated maps of them. Returns the state and the keys/values
    to carry: every position's keys/values of its last computed loop, unmapped."""
    attn = layer.self_attn
    b, t, _ = h.shape
    x = layer.input_layernorm(h)
    shape = (b, t, -1, attn.head_dim)
    q = attn.q_proj(x).view(shape).transpose(1, 2)
    k = attn.k_proj(x).view(shape).transpose(1, 2)
    v = attn.v_proj(x).view(shape).transpose(1, 2)
    q, k = rot(q, k, cos, sin)
    k_carry, v_carry = k, v
    if k_prev is not None:
        m = stale[:, None, :, None]
        k_carry = torch.where(m, k_prev, k)
        v_carry = torch.where(m, v_prev, v)
        if fill is None:
            k, v = k_carry, v_carry
        else:
            k_f, v_f = _mapped_fill(fill, k_carry, v_carry, cos, sin)
            k = torch.where(m, k_f, k)
            v = torch.where(m, v_f, v)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    a = attn.o_proj(a.transpose(1, 2).reshape(b, t, -1))
    h = h + layer.input_layernorm_2(a)
    h = h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
    return h, k_carry, v_carry


def routed_forward(model, ids, n_prompt, *, plan=None, thresholds=None, shares=None, lengths=None, exact_cache=False,
                   grad_ckpt=False, gate=None, fill_maps=None):
    """Teacher-forced forward of `ids` (B, T) under a router.

    n_prompt    (B,) number of prefill positions per sequence (they always run every loop)
    plan        (B, T) long, exit loop 0..3 of every position (prefill positions are forced to 3); or
    shares      [f1, f2, f3] (or a (B, 3) tensor): after loop s the top share f_{s+1} of the still-running answer tokens of
                each sequence, ranked by the gate's cumulative exit probability, leaves (needs `lengths`, (B,)); or
    thresholds  [q1, q2, q3]: a position leaves after loop s (0-based s = 0, 1, 2) when the gate's cumulative exit
                probability is >= q_{s+1}, exactly the decoder's rule; neither given = full depth
    exact_cache True = "gather" mode: exits are decided/recorded the same way but no key or value is replaced (every position
                keeps computing; the state at its exit loop is what an exit with an exact cache would read)
    gate        module mapping the normed state to the gate logit (default: the model's own early_exit_gate)
    fill_maps   (key maps, value maps) of load_fill_maps: calibrated fill instead of the plain copy
    Returns {"hidden": [4 x (B, T, D)] normed states, "exit": (B, T) long, "lam": (B, T, 4) float32 gate probabilities,
    "gate_logit": (B, T, 4) gate logits in the model's dtype}.
    hidden[t][b, i] is meaningful only for t <= exit[b, i] (always, in gather mode).
    """
    inner = model.model
    rot = _rotary(model)
    gate = gate if gate is not None else inner.early_exit_gate
    b, t_len = ids.shape
    dev = ids.device
    h = inner.embed_tokens(ids)
    pos = torch.arange(t_len, device=dev)[None].expand(b, t_len)
    cos, sin = inner.rotary_emb(h, pos)
    is_gen = pos >= n_prompt[:, None]
    last = N_LOOPS - 1
    if plan is not None:
        exit_ = torch.where(is_gen, plan.clamp(0, last), torch.full_like(pos, last))
    else:
        exit_ = torch.full_like(pos, last)
    stale = torch.zeros_like(is_gen)
    n_layers = len(inner.layers)
    ks, vs = [None] * n_layers, [None] * n_layers
    hidden, lams, glogs = [], [], []
    cum = torch.zeros((b, t_len), dtype=torch.float32, device=dev)
    rem = torch.ones_like(cum)
    for t in range(N_LOOPS):
        st = stale if (t > 0 and not exact_cache) else None
        for l, layer in enumerate(inner.layers):
            kp, vp = (ks[l], vs[l]) if st is not None else (None, None)
            fill = (exit_, t, l, fill_maps[0], fill_maps[1]) if (fill_maps is not None and st is not None) else None
            if grad_ckpt:
                h, ks[l], vs[l] = checkpoint(_block, layer, rot, h, cos, sin, kp, vp, st, fill, use_reentrant=False)
            else:
                h, ks[l], vs[l] = _block(layer, rot, h, cos, sin, kp, vp, st, fill)
        h = inner.norm(h)
        hidden.append(h)
        glog = gate(h).squeeze(-1)
        glogs.append(glog)
        lam = torch.sigmoid(glog.float())
        lams.append(lam)
        if t < last:
            if (thresholds is not None or shares is not None) and plan is None:
                with torch.no_grad():
                    cum = cum + lam.detach() * rem
                    rem = rem * (1.0 - lam.detach())
                    running = is_gen & (exit_ == last)
                    if shares is not None:
                        running = running & (pos < lengths[:, None])
                        newly = torch.zeros_like(running)
                        for bidx in range(b):
                            idx = running[bidx].nonzero(as_tuple=True)[0]
                            f = float(shares[bidx][t]) if torch.is_tensor(shares) else float(shares[t])
                            k = int(round(f * idx.numel()))
                            if k > 0:
                                top = torch.topk(cum[bidx, idx], k).indices
                                newly[bidx, idx[top]] = True
                    else:
                        newly = running & (cum >= float(thresholds[t]))
                    exit_ = torch.where(newly, torch.full_like(exit_, t), exit_)
            stale = is_gen & (exit_ <= t)
    return {"hidden": hidden, "exit": exit_, "lam": torch.stack(lams, dim=-1), "gate_logit": torch.stack(glogs, dim=-1)}


@torch.no_grad()
def stock_readout_loop(gate_logit, exit_):
    """The loop whose state the serving model's head actually reads for a position that ran loops 0..exit.

    The released forward keeps its stock selection on top of the decode-time exit: it reads the FIRST loop at which the
    cumulative exit probability, computed in the model's dtype (bfloat16), reaches the stock threshold 1.0, else the last
    computed loop. In bfloat16 the sigmoid of a gate logit above about 5.5 is exactly 1, so a token whose gate is that sure
    after loop s is read from loop s even when the router made it run further (its keys and values of the later loops are
    still the computed ones). With router thresholds below 1 such a token has left at loop s anyway; the rule matters for
    full depth and for routers that forbid an exit (threshold 1.0).
    """
    lam = torch.sigmoid(gate_logit)                       # model dtype, as in OuroForCausalLM.forward
    rem = torch.ones_like(lam[..., 0])
    cum = torch.zeros_like(rem)
    read = exit_.clone()
    found = torch.zeros_like(exit_, dtype=torch.bool)
    for t in range(N_LOOPS):
        cum = cum + lam[..., t] * rem
        rem = rem * (1.0 - lam[..., t])
        hit = (~found) & (cum >= 1.0) & (exit_ >= t)
        read = torch.where(hit, torch.full_like(read, t), read)
        found = found | hit
    return read


# ---------------------------------------------------------------------------------------------------------------- routers
def sample_iid_plan(ids, gen: torch.Generator, alpha: float = 1.0, rates=None):
    """The random router: every position draws its exit loop independently from a per-sequence distribution pi over the 4
    loops; pi ~ Dirichlet(alpha) per sequence (so a batch covers shallow, deep and mixed routers), or the fixed `rates`."""
    b, t_len = ids.shape
    if rates is not None:
        pi = torch.tensor(rates, dtype=torch.float32).expand(b, N_LOOPS)
    else:
        g = torch.distributions.Gamma(torch.full((b, N_LOOPS), float(alpha)), torch.ones(b, N_LOOPS))
        # Gamma draws through the default generator; seed it from ours so runs are reproducible
        state = torch.random.get_rng_state()
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
        x = g.sample()
        torch.random.set_rng_state(state)
        pi = x / x.sum(-1, keepdim=True)
    plan = torch.multinomial(pi, t_len, replacement=True, generator=gen)
    return plan.to(ids.device), pi


def sample_shares(b: int, gen: torch.Generator, alpha: float = 1.0):
    """Per-sequence exit shares for the "shares" router: pi ~ Dirichlet(alpha) over the 4 exit loops, turned into the share of
    the still-running tokens that leaves after loops 1, 2 and 3 (so the realised exit distribution is pi)."""
    state = torch.random.get_rng_state()
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    x = torch.distributions.Gamma(torch.full((b, N_LOOPS), float(alpha)), torch.ones(b, N_LOOPS)).sample()
    torch.random.set_rng_state(state)
    pi = x / x.sum(-1, keepdim=True)
    left = torch.ones(b)
    f = []
    for t in range(N_LOOPS - 1):
        f.append((pi[:, t] / left.clamp(min=1e-6)).clamp(0, 1))
        left = left - pi[:, t]
    return torch.stack(f, dim=-1), pi


def sample_thresholds(gen: torch.Generator, lo: float = 0.05, hi: float = 1.0):
    """A gate router with random per-loop thresholds, uniform in the search box of the router searches."""
    return [float(lo + (hi - lo) * torch.rand((), generator=gen)) for _ in range(N_LOOPS - 1)]


# ------------------------------------------------------------------------------------------------------------ loss, metrics
def _head_ce(head, x, tg):
    logits = head(x).float()
    return F.cross_entropy(logits, tg, reduction="none"), (logits.argmax(-1) == tg).float()


def readout_stats(model, out, ids, n_prompt, lengths, *, loss_on="valid", need_grad=True, loop_w=None):
    """Cross-entropy of the next-token readouts against the teacher-forced tokens.

    A readout position i (n_prompt - 1 <= i < length - 1) predicts ids[i + 1]. Readout (i, t) is valid when t <= exit[i].
    loss_on = "valid": mean over all valid readouts (every loop the token ran is asked for the full-depth token);
              "exit":  mean over the exit readouts only (what the decoder emits under this router).
    loop_w  per-loop weights of the "valid" loss (default 1, 1, 1, 1). The loop-1 readout never sees a filled cache slot (every
            position computes loop 1), so its loss only asks for a better first loop; the cache is in the readouts of loops 2-4.
    Returns (loss, stats) with per-loop sums for logging: n, ce and top-1 agreement of the valid readouts of every loop and of
    the exit readouts.
    """
    b, t_len = ids.shape
    dev = ids.device
    pos = torch.arange(t_len, device=dev)[None].expand(b, t_len)
    read = (pos >= (n_prompt[:, None] - 1)) & (pos < (lengths[:, None] - 1))
    bi, ti = read.nonzero(as_tuple=True)
    target = ids[bi, ti + 1]
    ex = out["exit"][bi, ti]
    stats = {"n_read": int(read.sum()), "loops_sum": float((ex + 1).sum())}
    total = ids.new_zeros((), dtype=torch.float32)
    denom = 0.0
    exit_ce = exit_ok = 0.0
    for t in range(N_LOOPS):
        valid = ex >= t
        if not bool(valid.any()):
            stats[f"n{t}"], stats[f"ce{t}"], stats[f"ok{t}"] = 0, 0.0, 0.0
            continue
        hv = out["hidden"][t][bi[valid], ti[valid]]
        tg = target[valid]

        if need_grad and hv.requires_grad:
            # the vocabulary-sized logits of a loop are recomputed in the backward pass instead of being kept for all loops
            # (the targets are passed as an argument: the function runs again in the backward pass, after this loop has moved on)
            ce, ok = checkpoint(_head_ce, model.lm_head, hv, tg, use_reentrant=False)
        else:
            ce, ok = _head_ce(model.lm_head, hv, tg)
        ok = ok.detach()
        at_exit = (ex[valid] == t).float()
        stats[f"n{t}"], stats[f"ce{t}"], stats[f"ok{t}"] = int(valid.sum()), float(ce.detach().sum()), float(ok.sum())
        exit_ce += float((ce.detach() * at_exit).sum())
        exit_ok += float((ok * at_exit).sum())
        if loss_on == "valid":
            w = 1.0 if loop_w is None else float(loop_w[t])
            total = total + w * ce.sum()
            denom += w * float(valid.sum())
        else:
            total = total + (ce * at_exit).sum()
            denom += float(at_exit.sum())
    stats["exit_ce"], stats["exit_ok"] = exit_ce, exit_ok
    loss = total / max(denom, 1.0)
    return (loss if need_grad else loss.detach()), stats


# -------------------------------------------------------------------------------------------------------------------- data
def load_teacher(path: str, split: str | None = None, max_len: int = 4096):
    """Rows of gen_teacher.py as {"question_id", "ids", "n_prompt", ...}; sequences longer than max_len are cut at the end.
    `path` may list several files (comma-separated, globs allowed); a problem generated twice is taken once."""
    import glob

    files = []
    for part in path.split(","):
        files += sorted(glob.glob(part)) if any(c in part for c in "*?[") else [part]
    seqs, seen = [], set()
    lines = [line for f in files for line in open(f)]
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:  # a line still being written by a generation process
            continue
        if r["question_id"] in seen:
            continue
        seen.add(r["question_id"])
        if split is not None and r.get("split") != split:
            continue
        ids = (r["prompt_ids"] + r["gen_ids"])[:max_len]
        if len(ids) - len(r["prompt_ids"]) < 2:
            continue
        seqs.append({"question_id": r["question_id"], "ids": ids, "n_prompt": len(r["prompt_ids"]), "difficulty": r.get("difficulty"),
                     "truncated": r.get("truncated", False)})
    return seqs


def make_batches(seqs, budget_tok: int, max_batch: int = 16, shuffle_gen: torch.Generator | None = None):
    """Batches of similar length with at most budget_tok padded tokens; shuffled order when a generator is given."""
    idx = list(range(len(seqs)))
    if shuffle_gen is not None:
        perm = torch.randperm(len(seqs), generator=shuffle_gen).tolist()
        # sort inside chunks of 64 so that neighbours have similar length but the epoch order is random
        idx = []
        for c in range(0, len(perm), 64):
            idx += sorted(perm[c:c + 64], key=lambda i: len(seqs[i]["ids"]))
    else:
        idx.sort(key=lambda i: len(seqs[i]["ids"]))
    batches, cur, cur_max = [], [], 0
    for i in idx:
        n = len(seqs[i]["ids"])
        if cur and (len(cur) >= max_batch or max(cur_max, n) * (len(cur) + 1) > budget_tok):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        batches.append(cur)
    if shuffle_gen is not None:
        order = torch.randperm(len(batches), generator=shuffle_gen).tolist()
        batches = [batches[j] for j in order]
    return batches


def collate(seqs, batch, pad_id: int, device: str = "cuda"):
    n = max(len(seqs[i]["ids"]) for i in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    for j, i in enumerate(batch):
        ids[j, :len(seqs[i]["ids"])] = torch.tensor(seqs[i]["ids"])
    n_prompt = torch.tensor([seqs[i]["n_prompt"] for i in batch])
    lengths = torch.tensor([len(seqs[i]["ids"]) for i in batch])
    return ids.to(device), n_prompt.to(device), lengths.to(device)


# -------------------------------------------------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update: y = W x + (alpha / r) B A x (B starts at zero)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale

    @torch.no_grad()
    def merged_weight(self):
        return (self.base.weight.float() + self.scale * (self.B @ self.A)).to(self.base.weight.dtype)


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def add_lora(model, r: int = 32, alpha: float = 64.0, targets=LORA_TARGETS):
    """Wrap the attention and feed-forward projections of every layer; everything else stays frozen. Returns the LoRA parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in targets:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    wrapped = LoRALinear(lin, r, alpha)
                    setattr(parent, name, wrapped)
                    params += [wrapped.A, wrapped.B]
    return params


def lora_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")}


def load_lora_state(model, state):
    own = dict(model.named_parameters())
    missing = [n for n in state if n not in own]
    assert not missing, f"LoRA keys not in the model: {missing[:3]}"
    with torch.no_grad():
        for n, v in state.items():
            own[n].copy_(v.to(own[n].device, own[n].dtype))


@torch.no_grad()
def merge_lora(model):
    """Fold every LoRA update into its base weight (in float32, stored back in the weight's dtype) and unwrap the layers."""
    n = 0
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LORA_TARGETS:
                mod = getattr(parent, name, None)
                if isinstance(mod, LoRALinear):
                    mod.base.weight.copy_(mod.merged_weight())
                    setattr(parent, name, mod.base)
                    n += 1
    return n
