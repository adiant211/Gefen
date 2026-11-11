"""Deterministic check that a Gefen copy reproduces another bit for bit on identical gradients.

usage: python equiv_check.py <gefen_src_dir> <out.pt>   (run once per copy, then compare with --compare a.pt b.pt)
Uses fixed tensors shaped like the Qwen3-100M parameters (period>1 and period-1 paths), 20 optimizer steps.
"""
import sys

import torch

if sys.argv[1] == "--compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    same = all(torch.equal(a[k], b[k]) for k in a)
    worst = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
    print("bitwise identical:", same, "| max abs diff:", worst)
    sys.exit(0 if same else 1)

sys.path.insert(0, sys.argv[1])
from gefen import Gefen

torch.manual_seed(0)
shapes = [(32000, 640), (640, 640), (320, 640), (1920, 640), (640, 1920), (640,), (64,)]
params = [torch.nn.Parameter(torch.randn(s, device="cuda") * 0.02) for s in shapes]
named = [(f"p{i}", p) for i, p in enumerate(params)]
opt = Gefen([{"params": named[:5], "weight_decay": 0.1}, {"params": named[5:], "weight_decay": 0.0}],
            lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
g = torch.Generator(device="cuda").manual_seed(1)
for step in range(20):
    for p in params:
        # structured gradients: per-row scale so the period search finds blocks
        scale = torch.rand(p.shape[0], *([1] * (p.dim() - 1)), device="cuda", generator=g) + 0.1
        p.grad = torch.randn(p.shape, device="cuda", generator=g) * scale * 1e-3
    opt.step()
out = {f"p{i}": p.detach().cpu() for i, p in enumerate(params)}
for i, p in enumerate(params):
    st = opt.state[p]
    out[f"codes{i}"] = st["m_codebook"].cpu()
    out[f"period{i}"] = torch.tensor(st["automatic_period"])
torch.save(out, sys.argv[2])
print("periods:", [opt.state[p]["automatic_period"] for p in params])
