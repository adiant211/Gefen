"""Qwen3-architecture ~100M model pretraining benchmark: AdamW vs Gefen (and Gefen variants).

Writes one JSONL log per run to --out/<opt>-<variant>-s<seed>.jsonl.
Gefen variants are selected with the GEFEN_FLAGS environment variable (read by the patched gefen package);
--variant is only the label used in the file name.

Update-error probe (--probe_every N): keeps exact fp32 AdamW moments ("shadow") on the same gradients and,
every N steps, compares the update the optimizer actually applied with the exact AdamW update.
For Gefen it also splits the error into a momentum-only and a second-moment-only part.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--opt", choices=["adamw", "gefen", "muon", "gefenmuon"], required=True,
                help="muon/gefenmuon: hidden 2D matrices use (Gefen)Muon, embedding and norms use fused AdamW")
ap.add_argument("--normuon", type=int, default=0, help="muon/gefenmuon: NorMuon per-row normalization")
ap.add_argument("--momentum", type=float, default=0.95, help="muon/gefenmuon momentum (Nesterov)")
ap.add_argument("--variant", default="base")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tokens", type=float, default=50e6)
ap.add_argument("--batch", type=int, default=64, help="sequences per optimizer step")
ap.add_argument("--micro", type=int, default=32, help="sequences per forward/backward")
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--beta1", type=float, default=0.9)
ap.add_argument("--beta2", type=float, default=0.95)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--warmup", type=float, default=0.02)
ap.add_argument("--min_lr_frac", type=float, default=0.1)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=100)
ap.add_argument("--eval_tokens", type=float, default=1e6)
ap.add_argument("--final_eval_tokens", type=float, default=4e6)
ap.add_argument("--probe_every", type=int, default=25, help="0 disables the update-error probe")
ap.add_argument("--precision", choices=["amp_bf16", "pure_bf16"], default="amp_bf16")
ap.add_argument("--compile", type=int, default=1)
ap.add_argument("--max_steps", type=int, default=0, help="stop early (throughput tests)")
ap.add_argument("--data", default="/workspace/data")
ap.add_argument("--out", default="/workspace/runs")
ap.add_argument("--gefen_src", default="/workspace/src")
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"
os.makedirs(args.out, exist_ok=True)
run_name = f"{args.opt}-{args.variant}-s{args.seed}"
log_path = os.path.join(args.out, run_name + ".jsonl")
if os.path.exists(log_path) and any('"type": "final"' in line for line in open(log_path)):
    print(f"{run_name}: already finished, skipping", flush=True)
    sys.exit(0)
log_f = open(log_path, "w")


def log(rec):
    log_f.write(json.dumps(rec) + "\n")
    log_f.flush()


meta = json.load(open(os.path.join(args.data, "meta.json")))
train_data = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

# ---------------------------------------------------------------- model
torch.manual_seed(args.seed)
cfg = Qwen3Config(
    vocab_size=meta["vocab"],
    hidden_size=640,
    intermediate_size=1920,
    num_hidden_layers=16,
    num_attention_heads=10,
    num_key_value_heads=5,
    head_dim=64,
    max_position_embeddings=args.seq,
    tie_word_embeddings=True,
    attention_bias=False,
    rms_norm_eps=1e-6,
    use_cache=False,
    bos_token_id=meta["eot"],
    eos_token_id=meta["eot"],
)
cfg._attn_implementation = "sdpa"
model = Qwen3ForCausalLM(cfg).to(dev)
if args.precision == "pure_bf16":
    model.to(torch.bfloat16)
n_params = sum(p.numel() for p in model.parameters())
named = list(model.named_parameters())
fwd = torch.compile(model) if args.compile else model
amp = args.precision == "amp_bf16"

# ---------------------------------------------------------------- optimizer
decay = [(n, p) for n, p in named if p.dim() >= 2]
no_decay = [(n, p) for n, p in named if p.dim() < 2]
betas = (args.beta1, args.beta2)
if args.opt == "adamw":
    opt = torch.optim.AdamW(
        [{"params": [p for _, p in decay], "weight_decay": args.wd},
         {"params": [p for _, p in no_decay], "weight_decay": 0.0}],
        lr=args.lr, betas=betas, eps=1e-8, fused=True)
elif args.opt == "gefen":
    sys.path.insert(0, args.gefen_src)
    from gefen import Gefen
    opt = Gefen(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=betas, eps=1e-8, weight_decay=args.wd, fused=True)
else:
    sys.path.insert(0, args.gefen_src)
    from gefen.gefen_muon import EPS as NS_EPS, DEFAULT_A, DEFAULT_B, DEFAULT_C, \
        _zeropower_via_newtonschulz, normuon_normalize
    NS_COEFFS = (DEFAULT_A, DEFAULT_B, DEFAULT_C)

    def muon_update(m_new, grad, row_v, lr, shape):
        """Exact (Nor)Muon update from new momentum m_new: Nesterov, Newton-Schulz, RMS 0.2 like AdamW."""
        u = m_new.mul(args.momentum).add_(grad, alpha=1 - args.momentum)
        o = _zeropower_via_newtonschulz(u, NS_COEFFS, 5, NS_EPS)
        if args.normuon:
            o = normuon_normalize(o, row_v, args.beta2)
            return o.mul_(lr * 0.2 * math.sqrt(o.numel()) / o.norm().clamp(min=1e-12))
        return o.float().mul_(lr * 0.2 * math.sqrt(max(shape[0], shape[1])))  # adjust_lr_fn="match_rms_adamw"

    class RefMuon(torch.optim.Optimizer):
        """Reference (Nor)Muon with an fp32 momentum buffer, same math as GefenMuon."""

        def __init__(self, params, lr, weight_decay):
            super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

        @torch.no_grad()
        def step(self):
            for g in self.param_groups:
                for p in g["params"]:
                    st = self.state[p]
                    if not st:
                        st["m"] = torch.zeros_like(p, dtype=torch.float32)
                        st["v"] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
                    grad = p.grad.float()
                    st["m"].lerp_(grad, 1 - args.momentum)
                    u = muon_update(st["m"], grad, st["v"], g["lr"], p.shape)
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.sub_(u)

    class Multi:
        """Steps several optimizers as one (Muon for hidden matrices, AdamW for the rest)."""

        def __init__(self, *opts):
            self.opts = opts
            self.param_groups = [g for o in opts for g in o.param_groups]

        @property
        def state(self):
            return {p: s for o in self.opts for p, s in o.state.items()}

        def step(self):
            for o in self.opts:
                o.step()

        def zero_grad(self, set_to_none=True):
            for o in self.opts:
                o.zero_grad(set_to_none=set_to_none)

    muon_named = [(n, p) for n, p in named if "layers." in n and p.dim() == 2]
    muon_set = {p for _, p in muon_named}
    rest_decay = [p for _, p in decay if p not in muon_set]
    aux = torch.optim.AdamW(
        [{"params": rest_decay, "weight_decay": args.wd},
         {"params": [p for _, p in no_decay], "weight_decay": 0.0}],
        lr=args.lr, betas=betas, eps=1e-8, fused=True)
    if args.opt == "muon":
        main = RefMuon([p for _, p in muon_named], lr=args.lr, weight_decay=args.wd)
    else:
        from gefen.gefen_muon import GefenMuon
        main = GefenMuon([{"params": muon_named, "weight_decay": args.wd}], lr=args.lr, weight_decay=args.wd,
                         momentum=args.momentum, nesterov=True, adjust_lr_fn="match_rms_adamw",
                         normuon=bool(args.normuon), normuon_beta2=args.beta2, fused=True)
    opt = Multi(main, aux)

# per-parameter lookup: name, weight decay
param_info = {}
for g in opt.param_groups:
    for p in g["params"]:
        param_info[p] = g
name_of = {p: n for n, p in named}


def category(name):
    for key in ("embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if key in name:
            return key
    return "norm"


# ---------------------------------------------------------------- data
total_steps = int(args.tokens // (args.batch * args.seq))
if args.max_steps:
    total_steps = min(total_steps, args.max_steps)
warmup_steps = max(1, int(args.warmup * total_steps))
accum = args.batch // args.micro
assert accum * args.micro == args.batch
rng = np.random.default_rng(args.seed + 12345)


def to_dev(arr):
    return torch.from_numpy(arr.astype(np.int64)).pin_memory().to(dev, non_blocking=True)


def train_batch():
    ix = rng.integers(0, len(train_data) - args.seq - 1, size=args.micro)
    t = to_dev(np.stack([train_data[i:i + args.seq + 1] for i in ix]))
    return t[:, :-1], t[:, 1:]


def lr_at(step):
    if step < warmup_steps:
        return args.lr * (step + 1) / warmup_steps
    frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def loss_fn(x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = fwd(input_ids=x).logits
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(n_tokens):
    model.eval()
    n_seq = int(n_tokens // args.seq)
    total, count = 0.0, 0
    for i in range(0, n_seq, args.micro):
        b = min(args.micro, n_seq - i)
        t = to_dev(np.stack([val_data[s * args.seq:s * args.seq + args.seq + 1] for s in range(i, i + b)]))
        total += loss_fn(t[:, :-1], t[:, 1:]).item() * b
        count += b
    model.train()
    return total / count


# ---------------------------------------------------------------- update-error probe
class Probe:
    def __init__(self):
        self.m = {p: torch.zeros_like(p, dtype=torch.float32) for _, p in named}
        self.v = {p: torch.zeros_like(p, dtype=torch.float32) for _, p in named}
        self.t = 0

    @torch.no_grad()
    def accumulate(self, measure=False):
        self.t += 1
        for _, p in named:
            g = p.grad.float()
            self.m[p].lerp_(g, 1 - args.beta1)
            self.v[p].mul_(args.beta2).addcmul_(g, g, value=1 - args.beta2)

    @torch.no_grad()
    def snapshot(self):
        return {p: p.detach().float().clone() for _, p in named}

    @torch.no_grad()
    def measure(self, before):
        bc1 = 1 - args.beta1 ** self.t
        bc2 = 1 - args.beta2 ** self.t
        sums = {}
        for _, p in named:
            g = param_info[p]
            lr, wd = g["lr"], g["weight_decay"]
            u_applied = before[p] * (1 - lr * wd) - p.detach().float()
            denom = (self.v[p] / bc2).sqrt().add_(1e-8)
            u_exact = lr * (self.m[p] / bc1) / denom
            rec = {"ref": u_exact.square().sum().item(),
                   "tot": (u_applied - u_exact).square().sum().item()}
            if args.opt == "gefen":
                st = opt.state[p]
                codes = st["m_codebook"]
                period = st["automatic_period"]
                m_g = opt._gefen_codebook[codes.long()].float() * opt._m_magnitude_view_for_m(st, codes).float()
                v_g = st["vmean"].float().expand(-1, period)
                factor = opt.v_profile_factor(st) if hasattr(opt, "v_profile_factor") else None
                if factor is not None:  # stored state lives in the profile-scaled space
                    m_g = m_g * factor.sqrt()
                    v_g = v_g * factor
                m_g = m_g.view_as(p)
                v_g = v_g.reshape(p.shape)
                u_m = lr * (m_g / bc1) / denom
                u_v = lr * (self.m[p] / bc1) / ((v_g / bc2).sqrt() + 1e-8)
                rec["mom"] = (u_m - u_exact).square().sum().item()
                rec["v"] = (u_v - u_exact).square().sum().item()
            for key in ("all", category(name_of[p])):
                acc = sums.setdefault(key, {})
                for k, val in rec.items():
                    acc[k] = acc.get(k, 0.0) + val
        out = {}
        for key, acc in sums.items():
            out[key] = {k: math.sqrt(acc[k] / acc["ref"]) for k in acc if k != "ref"}
        return out


class MuonProbe:
    """Shadow exact (Nor)Muon on the same gradients for the Muon-handled matrices.

    tot: applied update vs exact update. mom: exact update recomputed from the momentum the optimizer stored.
    """

    def __init__(self):
        self.m = {p: torch.zeros_like(p, dtype=torch.float32) for _, p in muon_named}
        self.v = {p: torch.zeros(p.shape[0], dtype=torch.float32, device=p.device) for _, p in muon_named}
        self.u, self.v_prev = {}, {}

    @torch.no_grad()
    def accumulate(self, measure=False):
        for _, p in muon_named:
            g = p.grad.float()
            self.m[p].lerp_(g, 1 - args.momentum)
            if measure:
                self.v_prev[p] = self.v[p].clone()
            u = muon_update(self.m[p], g, self.v[p], param_info[p]["lr"], p.shape)
            if measure:
                self.u[p] = u

    @torch.no_grad()
    def snapshot(self):
        return {p: p.detach().float().clone() for _, p in muon_named}

    @torch.no_grad()
    def measure(self, before):
        sums = {}
        for _, p in muon_named:
            gi = param_info[p]
            lr, wd = gi["lr"], gi["weight_decay"]
            u_applied = before[p] * (1 - lr * wd) - p.detach().float()
            u_exact = self.u.pop(p)
            rec = {"ref": u_exact.square().sum().item(), "tot": (u_applied - u_exact).square().sum().item()}
            if args.opt == "gefenmuon":
                st = main.state[p]
                codes = st["m_codebook"]
                m_g = (main._gefen_codebook[codes.long()].float()
                       * main._m_magnitude_view_for_m(st, codes).float()).view_as(p)
                u_m = muon_update(m_g, p.grad.float(), self.v_prev.pop(p), lr, p.shape)
                rec["mom"] = (u_m - u_exact).square().sum().item()
            for key in ("all", category(name_of[p])):
                acc = sums.setdefault(key, {})
                for k, val in rec.items():
                    acc[k] = acc.get(k, 0.0) + val
        return {key: {k: math.sqrt(acc[k] / acc["ref"]) for k in acc if k != "ref"} for key, acc in sums.items()}


if not args.probe_every:
    probe = None
elif args.opt in ("muon", "gefenmuon"):
    probe = MuonProbe()
else:
    probe = Probe()

# ---------------------------------------------------------------- train
log({"type": "config", "run": run_name, "args": vars(args), "n_params": n_params, "total_steps": total_steps,
     "gefen_flags": os.environ.get("GEFEN_FLAGS", ""), "torch": torch.__version__})
print(f"{run_name}: {n_params / 1e6:.2f}M params, {total_steps} steps, accum {accum}", flush=True)

t_start = time.time()
t_last = t_start
for step in range(total_steps):
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    loss_sum = 0.0
    for _ in range(accum):
        x, y = train_batch()
        loss = loss_fn(x, y) / accum
        loss.backward()
        loss_sum += loss.detach()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

    do_probe = probe is not None and (step % args.probe_every == 0 or step == total_steps - 1)
    if probe is not None:
        probe.accumulate(do_probe)
    before = probe.snapshot() if do_probe else None
    opt.step()
    if do_probe:
        log({"type": "probe", "step": step, "err": probe.measure(before)})
        del before
    opt.zero_grad(set_to_none=True)

    if step % 10 == 0 or step == total_steps - 1:
        torch.cuda.synchronize()
        now = time.time()
        n_steps = 10 if step else 1
        tok_s = n_steps * args.batch * args.seq / (now - t_last)
        t_last = now
        rec = {"type": "train", "step": step, "tokens": (step + 1) * args.batch * args.seq,
               "loss": loss_sum.item(), "lr": lr, "gnorm": gnorm.item(), "tok_s": tok_s,
               "elapsed": now - t_start}
        log(rec)
        print(f"step {step}/{total_steps} loss {rec['loss']:.4f} gnorm {rec['gnorm']:.2f} "
              f"{tok_s / 1e3:.1f}k tok/s", flush=True)
    if (step + 1) % args.eval_every == 0 and step != total_steps - 1:
        vl = evaluate(args.eval_tokens)
        log({"type": "eval", "step": step, "val_loss": vl})
        print(f"  val {vl:.4f}", flush=True)
        t_last = time.time()

final_val = evaluate(args.final_eval_tokens)
state_bytes = sum(v.numel() * v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
summary = {"type": "final", "val_loss": final_val, "state_bytes": state_bytes,
           "peak_mem": torch.cuda.max_memory_allocated(), "wall": time.time() - t_start}
if args.opt == "gefen":
    summary["periods"] = {name_of[p]: opt.state[p].get("automatic_period") for _, p in named}
if args.opt in ("muon", "gefenmuon"):
    summary["state_bytes_muon"] = sum(v.numel() * v.element_size() for s in main.state.values()
                                      for v in s.values() if torch.is_tensor(v))
    if args.opt == "gefenmuon":
        summary["periods"] = {name_of[p]: main.state[p].get("automatic_period") for _, p in muon_named}
log(summary)
print(f"FINAL {run_name}: val {final_val:.4f}, state {state_bytes / 2**20:.0f} MiB, "
      f"wall {summary['wall'] / 60:.1f} min", flush=True)
