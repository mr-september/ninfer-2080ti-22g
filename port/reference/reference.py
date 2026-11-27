"""FP32 single-visit decode reference for Inkling-2-Mini (M2).

Authoritative math: the reference server source E:/ninfer-all/repo/src/models/inkling2
(executor.cpp + kernels.cu), cross-checked by port/specs/*. Math is quoted from that
source as the reference is written; no semantics is copied from the port's C++ yet to
exist. Weights are the FP32 dequantizations of the source GGUF — the same decoded
values the running server computes on, so agreement with the baseline is a semantics
gate, not a weights comparison.

Deliberate mirror of server rounding (so greedy text can match): mix/controller
projection inputs -> BF16, attention output -> BF16, KV cache -> the engine's KV
coding: engine default "bf16"/"fp16" == plain F16 cache (kv_format.cpp:13); production
server runs int8 Int8G64 (INK2_KV_CODING=int8 models it; fp16|fp32 for diagnosis).
Server-only
quantized activations (q8_1 decode rows, fixed-point expert sums) are NOT mirrored;
they are bounded noise, and any *early* divergence remains a semantics bug.
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.convert.sources.gguf import GGUFSource  # noqa: E402

D = 128          # head_dim
REL = 16         # relative feature width
H = 2304         # hidden
Q_HEADS, KV_HEADS = 24, 8
GROUP = Q_HEADS // KV_HEADS
Q, KVD = Q_HEADS * D, KV_HEADS * D
W = 1152         # expert width
E = 128          # experts
TOPK = 8
EPS = 1e-6       # rms_norm_eps
MIX_EPS = 1e-6   # kMixEps
K_REL_EXTENT = 1024
SOFTCAP = 30.0
GATE_CAP, UP_CAP = 4.0, 25.0
ROUTE_SCALE, ROUTE_EPS = 8.0, 1e-20
N_SCORED = 200058
QK_CLIP = 22.627417
REL_CLIP = 8.0
import os as _os
_KV_CODING = _os.environ.get("INK2_KV_CODING", "int8")  # int8|fp16|fp32

def _int8_g64(x):
    """Engine Int8G64 round-trip (int8_g64_codec.cuh): per 64-dim group,
    scale = FP16-RNE(absmax/127), code = RN(x/fp16scale) clamp +-127, stored code*scale."""
    g = x.reshape(*x.shape[:-1], x.shape[-1] // 64, 64)
    s = (g.abs().amax(-1, keepdim=True) / 127.0).to(torch.float16).float()
    code = (g / s).round().clamp(-127, 127)
    return (code * s).reshape(x.shape)

def _kv_store(kc, vc, full):
    if _KV_CODING == "int8" and full:      # only full-attention visits use planes
        return _int8_g64(kc), _int8_g64(vc)
    dt = torch.float32 if _KV_CODING == "fp32" else torch.float16
    return kc.to(dt), vc.to(dt)
LOG_FLOOR, LOG_ALPHA = 8192.0, 0.1


ACT_MIRROR = __import__("os").environ.get("INK2_ACT_MIRROR", "0") == "1"


def q8_1_rt(x):
    """Round-trip rows through q8_1 (32-block, d = amax/127 stored fp16,
    codes rounded away from zero, dequantized fp16 d) - the engine's
    quantize_vector_value in ops/linear/gguf/ggml_bridge.cu."""
    shp = x.shape
    K = shp[-1]
    xb = x.reshape(-1, K // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    d = (amax / 127.0).half().float()
    q = xb / d.clamp_min(1e-30)
    code = torch.sign(q) * torch.floor(q.abs() + 0.5)
    code = torch.where(amax == 0, torch.zeros_like(code), code)
    return (code * d).reshape(shp)


def f16(x):
    return x.half().float()


def act_plane(x, t, from_f32=False):
    """Engine activation forms for a projection input plane: t <= 8 rows are
    quantized to q8_1 (from BF16, or FP32 directly at layer 0's rms output);
    longer calls read BF16."""
    if not ACT_MIRROR:
        return x
    if t <= 8:
        return q8_1_rt(x if from_f32 else bf16(x))
    return bf16(x)


def act_moe(x, t):
    """Expert gemm inputs: q8_1 rows for t <= 8 (vector kernel), F16 planes on
    sm_70-75 matrix calls (moe_sm70_activation_column)."""
    if not ACT_MIRROR:
        return x
    return q8_1_rt(bf16(x).reshape(1, -1)).reshape(x.shape) if t <= 8 else f16(x)


def bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def rms(v, weight, eps=EPS):
    """RMSNorm of rows of v (no weight if weight is None)."""
    n = torch.rsqrt((v * v).mean(-1, keepdim=True) + eps)
    out = v * n
    return out * weight if weight is not None else out


class Experts:
    """FP32 decode of selected experts only: expert e of the blob is a contiguous,
    block-aligned element range, decoded on demand through the GGUF source window."""

    CACHE = OrderedDict()          # (layer, e) -> (gate_up [2W,H], down [H,W])
    CACHE_MAX = 32
    GU_ELEMS, DOWN_ELEMS = 2 * W * H, H * W

    def __init__(self, store, layer):
        self.store, self.layer = store, layer
        self._gu = self._down = None

    def _pair(self, e):
        key = (self.layer, e)
        if key not in Experts.CACHE:
            if self._gu is None:
                self._gu = self.store.tensor(f"blk.{self.layer}.ffn_gate_up_exps.weight")
                self._down = self.store.tensor(f"blk.{self.layer}.ffn_down_exps.weight")
            gu = self._gu.values(e * Experts.GU_ELEMS, (e + 1) * Experts.GU_ELEMS).view(2 * W, H)
            dn = self._down.values(e * Experts.DOWN_ELEMS, (e + 1) * Experts.DOWN_ELEMS).view(H, W)
            gu = gu.to(torch.float32)
            dn = dn.to(torch.float32)
            Experts.CACHE[key] = (gu, dn)
            while len(Experts.CACHE) > Experts.CACHE_MAX:
                Experts.CACHE.popitem(last=False)
        return Experts.CACHE[key]

    def gate_up(self, e):
        return self._pair(e)[0]  # rows [0,W) gate, [W,2W) up (M2 gate)

    def down(self, e):
        return self._pair(e)[1]


class VisitState:
    """Everything position-indexed that a visit keeps across calls."""

    def __init__(self):
        self.cache = {}        # position -> (K F16 [KV,D], V F16 [KV,D])
        self.k_raw = {}        # position -> raw K projection [KVD] (conv input)
        self.v_raw = {}        # position -> raw V projection [KVD]
        self.y_attn = {}       # position -> raw attention-output projection [H]
        self.y_moe = {}        # position -> raw MoE-output projection [H]


class Inkling2Reference:
    def __init__(self, gguf_path, device="cpu"):
        self.store = GGUFSource(gguf_path)
        self.device = device
        meta = self.store.metadata
        self.experts_cache = {}
        self.visits = self._build_schedule()
        self.vstate = [VisitState() for _ in self.visits]
        self.w = {}  # per-layer decoded weights, built lazily

    # ---------------------------------------------------------------- weights
    def tensor(self, name):
        if name not in self.w:
            src = self.store.tensor(name)
            self.w[name] = src.values().reshape(src.shape).to(torch.float32)
        return self.w[name]

    def layer_weights(self, n):
        key = f"L{n}"
        if key not in self.w:
            p = f"blk.{n}."
            full = n in (5, 11, 17, 23)
            self.w[key] = {
                "full": full,
                "qgr": torch.cat([self.tensor(p + "attn_q.weight"),
                                  self.tensor(p + "attn_gate.weight"),
                                  self.tensor(p + "attn_rel.weight")], 0),
                "kv": torch.cat([self.tensor(p + "attn_k.weight"),
                                 self.tensor(p + "attn_v.weight")], 0),
                "out": self.tensor(p + "attn_output.weight"),
                "output_conv": self.tensor(p + "attn_post_conv.weight"),   # [H,4]
                "moe_conv": self.tensor(p + "ffn_post_conv.weight"),       # [H,4]
                "key_conv": self.tensor(p + "attn_k_conv.weight"),         # [KVD,4]
                "value_conv": self.tensor(p + "attn_v_conv.weight"),       # [KVD,4]
                "query_clip": self.tensor(p + "attn_q_clip.weight"),       # [heads,D]
                "relative_clip": self.tensor(p + "attn_rel_clip.weight"),  # [heads,REL]
                "distances": self.tensor(p + "attn_rel_bias.weight"),      # [1024,REL]
                "sinks": self.tensor(p + "attn_sink_proj.weight"),         # [heads,H]
                "attn_norm": self.tensor(p + "attn_norm.weight") if not full else None,
                "q_inp_norm": self.tensor(p + "attn_q_inp_norm.weight") if full else None,
                "k_inp_norm": self.tensor(p + "attn_k_inp_norm.weight") if full else None,
                "v_inp_norm": self.tensor(p + "attn_v_inp_norm.weight") if full else None,
                "key_taps": self.tensor(p + "attn_k_conv_proj.weight") if full else None,
                "value_taps": self.tensor(p + "attn_v_conv_proj.weight") if full else None,
                "attn_mix": self.tensor(p + "attn_res_query.weight") if n > 0 else None,
                "ffn_mix": self.tensor(p + "ffn_res_query.weight").reshape(1, H),  # [1,H]
                "ffn_norm": self.tensor(p + "ffn_norm.weight"),
                "loop_queries": self.tensor(p + "loop_msg_query.weight") if n in range(6, 18) else None,
                "router": self.tensor(p + "ffn_gate_inp.weight"),          # [E,H]
                "bias": self.tensor(p + "exp_probs_b.bias"),               # [E]
                "experts": Experts(self.store, n),
            }
            w = self.w[key]
            qm = w["attn_mix"]
            if qm is not None:
                w["attn_mix"] = qm.reshape(1, H) if qm.ndim == 1 else qm.contiguous()  # [nq,H]
            if w["loop_queries"] is not None:
                w["loop_queries"] = w["loop_queries"].contiguous()  # logical [4,H]
            for stream in ("q", "k", "v", "ffn"):
                if n in range(6, 18):
                    w[f"ctrl_{stream}_down"] = self.tensor(p + f"loop_{stream}_ctrl_down.weight")    # [64,2H]
                    w[f"ctrl_{stream}_forget"] = self.tensor(p + f"loop_{stream}_ctrl_forget.weight")  # [H,64]
                    w[f"ctrl_{stream}_gain"] = self.tensor(p + f"loop_{stream}_ctrl_gain.weight")      # [H,64]
        return self.w[key]

    @staticmethod
    def _build_schedule():
        visits = [(i, 0) for i in range(0, 6)]
        for k in range(3):
            visits += [(i, k) for i in range(6, 18)]
        visits += [(i, 0) for i in range(18, 24)]
        out = []
        for layer, ps in visits:
            out.append({
                "layer": layer, "pass": ps,
                "control": 6 <= layer < 18 and ps > 0,
                "start": layer % 6 == 0,
            })
        return out

    # ------------------------------------------------------------- ops (math)
    @staticmethod
    def mix(states, queries, norms, t):
        """states: list of [T,H] fp32; returns (raw [nq][T,H], normed [nq][T,H]).
        s_j = rms(state_j, no weight); weights softmax_j(q . s_j / sqrt(H));
        mixed = sum_j w_j s_j; normed = rms(mixed, norm[q])."""
        s = torch.stack([rms(x, None, MIX_EPS) for x in states])          # [J,T,H]
        scores = torch.einsum("qk,jtk->qtj", queries, s) * (1.0 / math.sqrt(H))
        w = torch.softmax(scores, dim=-1)                                  # [T,nq,J]
        raw = torch.einsum("qtj,jtk->qtk", w, s)
        normed = torch.stack([rms(raw[q], norms[q] if norms is not None else None)
                              for q in range(raw.shape[0])])
        return raw, normed

    @staticmethod
    def control(x, message, down, forget, gain, norm):
        """One controller row set: concat of the two RMSed planes, rank-64 blend.
        y = x - tanh(forget) x + tanh(gain) message, then RMS with norm."""
        concat = torch.cat([rms(x, None, MIX_EPS), rms(message, None, MIX_EPS)], -1)
        state = concat @ down.t()                                          # [T,64]
        f = state @ forget.t()                                             # [T,H]
        g = state @ gain.t()                                               # [T,H]
        y = x - torch.tanh(f) * x + torch.tanh(g) * message
        return rms(y, norm)

    @staticmethod
    def conv(static, rows, taps=None):
        """Short conv: x(p) + sum_i w[i] x(p-3+i) (+ sum_lag dt[lag] x(p-lag)).
        rows[i] = x(p - i) (i=0 own); static [C,4]; taps [4,C] per-lag or None."""
        out = rows[0] + sum(static[:, i] * rows[3 - i] for i in range(4))
        if taps is not None:
            out = out + sum(taps[lag] * rows[lag] for lag in range(4))
        divisor = (1.0 / math.sqrt(9.0)) if taps is not None else (1.0 / math.sqrt(5.0))
        return out * divisor

    # ---------------------------------------------------------------- forward
    def __call__(self, ids):
        """Prefill + one greedy continuation call: run all visits over the new
        token (or prompt block); return next-token logits [V]."""
        t = len(ids)
        first = self.pos if hasattr(self, "pos") else 0
        positions = list(range(first, first + t))
        self.pos = first + t

        # State pool: blocks (list of [T,H]) and cur, per the executor's States.
        x = self.tensor("token_embd.weight")[ids].clone()                  # [T,H]
        buffers = {0: rms(x, self.tensor("token_embd_norm.weight"))}       # s.cur = 0
        nxt = 1
        st = {"blocks": [], "cur": 0, "prelude_blocks": None, "prelude_cur": None,
              "bank": None, "bank_last": None}
        logits = None

        for v, vp in enumerate(self.visits):
            import os
            if os.environ.get("INK2_DEBUG"):
                stt = buffers[st["cur"]][-1]
                print(f"v{v:02d} L{vp['layer']:02d}p{vp['pass']} "
                      f"rms={float(stt.norm()/math.sqrt(H)):.6f} "
                      f"blocks={len(st['blocks'])}", flush=True)
            if getattr(self, "snap", None) is not None:
                self.snap.append(buffers[st["cur"]][getattr(self, "snap_row", -1)].clone())
            lp = self.layer_weights(vp["layer"])
            if vp["layer"] == 6 and vp["pass"] == 0:
                st["prelude_blocks"], st["prelude_cur"] = list(st["blocks"]), st["cur"]
            elif vp["layer"] == 6 and vp["control"]:
                st["bank"] = list(st["blocks"][len(st["prelude_blocks"]) + 1:])
                st["bank_last"] = st["cur"]
                st["blocks"], st["cur"] = list(st["prelude_blocks"]), st["prelude_cur"]

            states = [buffers[b] for b in st["blocks"]] + [buffers[st["cur"]]]

            if vp["control"]:
                mq = lp["loop_queries"]
                bank = [buffers[b] for b in st["bank"]] + [buffers[st["bank_last"]]]
                messages = self.mix(bank, mq, None, t)[0]                  # [4,T,H] raw
            else:
                messages = None

            # --- attention inputs
            if vp["layer"] == 0:
                plane = rms(buffers[st["cur"]], lp["attn_norm"])
                plane = act_plane(plane, t, from_f32=True)
                xin_f32 = [plane] * 3
            else:
                nq = lp["attn_mix"].shape[0]
                norms = ([lp["q_inp_norm"], lp["k_inp_norm"], lp["v_inp_norm"]]
                         if lp["full"] else [lp["attn_norm"]] * 3)
                raw_a = self.mix(states, lp["attn_mix"], norms, t)
                if vp["control"]:
                    xin_f32 = [act_plane(self.control(raw_a[0][i if lp["full"] else 0], messages[i],
                                            lp[f"ctrl_{'qkv'[i]}_down"],
                                            lp[f"ctrl_{'qkv'[i]}_forget"], lp[f"ctrl_{'qkv'[i]}_gain"],
                                            norms[i]), t) for i in range(3)]
                elif lp["full"]:
                    xin_f32 = [act_plane(raw_a[1][i], t) for i in range(3)]
                else:
                    xin_f32 = [act_plane(raw_a[1][0], t)] * 3
            q0 = xin_f32[0]

            # --- projections
            qgr = q0 @ lp["qgr"].t()                       # [T, 6528]
            kv = xin_f32[1] @ lp["kv"].t()                 # [T, 2*KVD]
            sinks = q0 @ lp["sinks"].t()                   # [T, heads]
            vs = self.vstate[v]
            full = lp["full"]

            tau = None
            if full:
                p = torch.tensor([float(p) for p in positions], device=q0.device)
                tau = 1.0 + LOG_ALPHA * torch.log(torch.clamp_min((p + 1.0) / LOG_FLOOR, 1.0))
                k_taps_all = xin_f32[1] @ lp["key_taps"].t()      # [T, KV*4] per (head, lag)
                v_taps_all = xin_f32[2] @ lp["value_taps"].t()

            # --- query_post (executor: query/r clipped per head, tau on full layers)
            qs = qgr[:, :Q].reshape(t, Q_HEADS, D) * lp["query_clip"]
            qf = QK_CLIP / (qs.norm(dim=-1, keepdim=True) + 1e-6)
            qs = qs * qf.clamp(max=1.0)
            rs = qgr[:, Q + Q:].reshape(t, Q_HEADS, REL) * lp["relative_clip"]
            rf = REL_CLIP / (rs.norm(dim=-1, keepdim=True) + 1e-6)
            rs = rs * rf.clamp(max=1.0)
            if full:
                qs = qs * tau[:, None, None]
                rs = rs * tau[:, None, None]

            # --- kv_post: conv on raw projections, K clip, V RMS, store cache + history
            new_raw = {}
            for i in range(t):
                pos = positions[i]
                lags_k = [kv[i, :KVD]]
                lags_v = [kv[i, KVD:]]
                for lag in (1, 2, 3):
                    src = (kv[i - lag, :KVD] if i - lag >= 0
                           else vs.k_raw.get(pos - lag, torch.zeros(KVD)))
                    lags_k.append(src)
                    src = (kv[i - lag, KVD:] if i - lag >= 0
                           else vs.v_raw.get(pos - lag, torch.zeros(KVD)))
                    lags_v.append(src)
                kt = v_taps = None
                if full:
                    kdt = k_taps_all[i].reshape(KV_HEADS, 4)
                    vdt = v_taps_all[i].reshape(KV_HEADS, 4)
                    kt = torch.stack([kdt[:, lag].repeat_interleave(D) for lag in range(4)])
                    v_taps = torch.stack([vdt[:, lag].repeat_interleave(D) for lag in range(4)])
                kc = self.conv(lp["key_conv"], lags_k, kt)
                vc = self.conv(lp["value_conv"], lags_v, v_taps)
                kc = kc.reshape(KV_HEADS, D)
                kf = (QK_CLIP / (kc.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)
                kc = kc * kf
                vc = vc.reshape(KV_HEADS, D)
                vc = vc * torch.rsqrt((vc * vc).mean(-1, keepdim=True) + EPS)
                kc_s, vc_s = _kv_store(kc, vc, full)
                vs.cache[pos] = (kc_s, vc_s)
                vs.k_raw[pos] = kv[i, :KVD].clone()
                vs.v_raw[pos] = kv[i, KVD:].clone()

            # --- attention (window keys d<1024; full keys all; bias d<1024; sink softmax)
            cache_keys = sorted(vs.cache)
            attn = torch.zeros(t, Q, device=q0.device)
            gate = qgr[:, Q:Q + Q]
            for i in range(t):
                pos = positions[i]
                lo = 0 if full else max(0, pos - 1023)
                keys = [p for p in cache_keys if lo <= p <= pos]
                dists = torch.tensor([pos - p for p in keys], device=q0.device)
                K = torch.stack([vs.cache[p][0].float() for p in keys])   # [n,KV,D]
                V = torch.stack([vs.cache[p][1].float() for p in keys])
                for h in range(Q_HEADS):
                    kh = h // GROUP
                    s = (K[:, kh] @ qs[i, h]) * (1.0 / math.sqrt(D))      # [n]
                    biased = dists < 1024
                    if biased.any():
                        s = s.index_add(0, biased.nonzero().squeeze(-1),
                                        lp["distances"][dists[biased]] @ rs[i, h])
                    m = s.max()
                    e = torch.exp(s - m)
                    sink = sinks[i, h]
                    top = torch.clamp_min(m, sink)
                    den = e.sum() * torch.exp(m - top) + torch.exp(sink - top)
                    attn[i, h * D:(h + 1) * D] = (e @ V[:, kh]) * (1.0 / den)
            attn = attn.reshape(t, Q_HEADS, D) * torch.sigmoid(gate.reshape(t, Q_HEADS, D))
            y = act_plane(bf16(attn.reshape(t, Q)), t) @ lp["out"].t()       # [T,H] fp32

            # --- attention residual (start pushes block) + ffn input
            if vp["start"]:
                st["blocks"].append(st["cur"])
                st["cur"] = nxt
                buffers[nxt] = torch.zeros(t, H, device=q0.device)
                nxt += 1
            cur = buffers[st["cur"]]
            c = torch.zeros(t, H, device=q0.device)
            for i in range(t):
                pos = positions[i]
                lags = [y[i]]
                for lag in (1, 2, 3):
                    lags.append(y[i - lag] if pos - lag >= first and i - lag >= 0
                                else vs.y_attn.get(pos - lag, torch.zeros(H)))
                c[i] = self.conv(lp["output_conv"], lags)
            if vp["start"]:
                cur.copy_(c)
            else:
                cur += c
            for i in range(t):
                vs.y_attn[positions[i]] = y[i].clone()

            ffn_states = [buffers[b] for b in st["blocks"]] + [buffers[st["cur"]]]
            raw_f, normed_f = self.mix(ffn_states, lp["ffn_mix"], [lp["ffn_norm"]], t)
            if vp["control"]:
                xffn = self.control(raw_f[0], messages[3], lp["ctrl_ffn_down"],
                                    lp["ctrl_ffn_forget"], lp["ctrl_ffn_gain"], lp["ffn_norm"])
            else:
                xffn = normed_f[0]

            # --- MoE: sigmoid router, top-8, SiTU, weighted down sum
            moe = torch.zeros(t, H, device=q0.device)
            for i in range(t):
                logits_r = torch.sigmoid(lp["router"] @ xffn[i])          # [E]
                pick = (logits_r + lp["bias"]).topk(TOPK)
                ids_sel = pick.indices
                p_sel = logits_r[ids_sel]
                weights = p_sel / (p_sel.sum() + ROUTE_EPS) * ROUTE_SCALE
                xrow = act_moe(xffn[i], t)
                for j, e in enumerate(ids_sel.tolist()):
                    ex = lp["experts"]
                    gu = ex.gate_up(e)
                    g = xrow @ gu[:W].t()
                    u = xrow @ gu[W:].t()
                    mid = (GATE_CAP * torch.tanh(g / GATE_CAP) * torch.sigmoid(g)
                           * UP_CAP * torch.tanh(u / UP_CAP))
                    moe[i] += weights[j] * (act_moe(bf16(mid), t) @ ex.down(e).t())

            # --- MoE residual + next mix (recomputed at the next visit here)
            c2 = torch.zeros(t, H, device=q0.device)
            for i in range(t):
                pos = positions[i]
                lags = [moe[i]]
                for lag in (1, 2, 3):
                    lags.append(moe[i - lag] if pos - lag >= first and i - lag >= 0
                                else vs.y_moe.get(pos - lag, torch.zeros(H)))
                c2[i] = self.conv(lp["moe_conv"], lags)
            cur += c2
            for i in range(t):
                vs.y_moe[positions[i]] = moe[i].clone()

            if v + 1 == len(self.visits):
                out_states = [buffers[b] for b in st["blocks"]] + [buffers[st["cur"]]]
                _, o = self.mix(out_states, self.tensor("output_res_query.weight").reshape(1, H),
                                [self.tensor("output_norm.weight")], t)
                logits = act_plane(o[0], t) @ self.tensor("output.weight")[:N_SCORED].t()
                logits = SOFTCAP * torch.tanh(logits / SOFTCAP)
        return logits[-1]

    def reset(self):
        self.vstate = [VisitState() for _ in self.visits]
        self.pos = 0

    def generate(self, ids, max_new):
        self.reset()
        out = list(ids)
        next_logits = self(out)
        for _ in range(max_new):
            tok = int(next_logits.argmax())
            out.append(tok)
            next_logits = self([tok])
        return out
