"""
Benchmark for neural solvers of unsteady singularly perturbed
convection-diffusion-reaction problems on the unit square:

    u_t - D Lap(u) + b.grad(u) + c u + g(u) = f   in (0,1)^2 x (0,T]
    u = 0 on the boundary,  u(.,0) = 0             (true for all problems here)

Problems  (eps = singular perturbation parameter)
  conv_mms   Ex. 1 (manufactured, exact solution)
  conv_var   Ex. 2 (variable velocity, no exact solution -> FD reference)
  react_mms  Ex. 3 (D = eps^2, manufactured, exact solution)
  react_var  Ex. 4: D = eps^2, variable reaction c = 1+3x+y (FD reference);
             true layer widths eps/sqrt(c) differ from eps on every side.

Methods (identical data, budget and optimiser); the full list is in SPECS below
  ae_hard_b   BLC-PINN: layer coordinates (rate 1/eps) + exact BC/IC (Coons projector x t)
              + layer-balanced residual weight
  ae_coef_b   BLC-PINN with coefficient-aware rates (Ex. 2, Ex. 4)
  ae_soft, ae_hard, ae_coef, ae_cc, ae_plus
              ablation variants (penalty BC; exact BC; coefficient-aware rates;
              corner features; trainable rate multipliers)
  pinn        standard PINN, penalty BC/IC
  fourier_s1  random Fourier-feature PINN (Tancik et al. 2020), sigma = 1, penalty BC/IC
  gkpinn_pub  GKPINN (Wang et al. 2024) with its published settings: N0 + sum_i N_i * phi_i
  tsnn_soft   two-scale NN (Zhuang et al. 2025) with its published treatment of boundary
              layers: inputs eps^{-1/2}(x-1/2), eps^{-1/2}(y-1/2), eps^{-1/2}; penalty BC/IC
  Other entries of SPECS (e.g. tsnn, hu_wl, papinn) are variants that were tested during the
  study but are not reported in the paper.

usage:
  python bench2.py --selftest                  # 1-2 min correctness checks
  python bench2.py --plan S2 --out DIR          # run the plan (resumable)
"""
import argparse, json, math, os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float32


# ============================================================ problems
class Problem:
    def __init__(self, name, eps):
        self.name, self.eps = name, eps
        e = eps
        if name == "conv_mms":
            self.D, self.T = e, 1.0
            self.sides = ["x1", "y1"]
            self.rate_orig = {"x1": 1 / e, "y1": 1 / e}
        elif name == "conv_var":
            self.D, self.T = e, 2.0
            self.sides = ["x1", "y1"]
            self.rate_orig = {"x1": 1 / e, "y1": 1 / e}
        elif name == "react_mms":
            self.D, self.T = e * e, 1.0
            self.sides = ["x0", "x1", "y0", "y1"]
            self.rate_orig = {s: 1 / e for s in self.sides}
        elif name == "react_var":
            self.D, self.T = e * e, 1.0
            self.sides = ["x0", "x1", "y0", "y1"]
            self.rate_orig = {s: 1 / e for s in self.sides}
        else:
            raise ValueError(name)
        self.has_exact = name.endswith("_mms")
        self.eval_times = [0.25 * self.T, 0.5 * self.T, self.T]

    # ---- coefficients (torch)
    def b(self, x, y):
        if self.name == "conv_mms":
            return torch.ones_like(x), torch.ones_like(y)
        if self.name == "conv_var":
            return 1 - x * y / 2, 1 + x * y / 2
        return None

    def c(self, x, y):
        if self.name == "react_var":
            return 1 + 3 * x + y
        return None

    def g(self, u):
        if self.name == "conv_mms":
            return u * torch.exp(u)
        if self.name == "conv_var":
            return u ** 2
        if self.name == "react_mms":
            return (u - 4) / (5 - u)
        return None

    def exact(self, x, y, t, lib=torch):
        e = self.eps
        if self.name == "conv_mms":
            a1 = -math.exp(-1 / e); a2 = -1 - a1
            P = lambda s: a1 + a2 * s + lib.exp(-(1 - s) / e)
            return (1 - lib.exp(-t)) * P(x) * P(y)
        if self.name == "react_mms":
            nrm = 1 + math.exp(-1 / e)
            P = lambda s: 1 - (lib.exp(-s / e) + lib.exp(-(1 - s) / e)) / nrm
            return (1 + t - lib.exp(-t)) * P(x) * P(y)
        return None

    def f_given(self, x, y, t):
        if self.name == "conv_var":
            return t ** 2 * (x * (1 - x) + y * (1 - y))
        if self.name == "react_var":
            return (1 - torch.exp(-5 * t)) * (1 + x * y)
        return None

    # ---- coefficient-aware layer rates (used by ae_plus)
    def rate_coef(self, side, x, y):
        e = self.eps
        if self.name == "conv_var":
            if side == "x1":
                return (1 - y / 2) / e          # b1(1,y)/eps
            if side == "y1":
                return (1 + x / 2) / e          # b2(x,1)/eps
        if self.name == "react_var":
            cc = {"x0": 1 + y, "x1": 4 + y, "y0": 1 + 3 * x, "y1": 2 + 3 * x}[side]
            return torch.sqrt(cc) / e           # sqrt(c on that edge)/eps
        # conv_mms: b=(1,1) -> 1/eps ; react_mms: manufactured with 1/eps,
        # no physically meaningful coefficient (nonlinear) -> keep 1/eps
        return torch.full_like(x, 1 / e)

    def balance_weight(self, x, y):
        """Layer-balanced residual weight w = 1/(1 + sum_i (A_i - 1) phi_i).
        Inside a layer the network's derivatives through the layer coordinate
        phi_i = exp(-r_i d_i) carry the amplification A_i = max(D r_i^2, |b.n_i| r_i)
        (= O(1/eps) for convection-diffusion, = O(1) for D = eps^2 reaction-diffusion);
        w rescales the residual there to O(1) and leaves the outer region unchanged."""
        tot = torch.zeros_like(x)
        bb = self.b(x, y)
        for s in self.sides:
            r = self.rate_coef(s, x, y)
            A = self.D * r ** 2
            if bb is not None:
                bn = bb[0] if s[0] == "x" else bb[1]
                A = torch.maximum(A, bn.abs() * r)
            A = torch.clamp(A, min=1.0)
            tot = tot + (A - 1) * torch.exp(-r * dist(s, x, y))
        return (1.0 / (1.0 + tot)).detach()

    def layer_width(self):
        """widest layer, for sampling and evaluation grids"""
        if self.name == "conv_var":
            return 2 * self.eps
        return self.eps


def dist(side, x, y):
    return {"x0": x, "x1": 1 - x, "y0": y, "y1": 1 - y}[side]


# ============================================================ networks
def mlp(n_in, width, depth, n_out=1, act=nn.Tanh):
    layers, d = [], n_in
    for _ in range(depth):
        lin = nn.Linear(d, width); nn.init.xavier_normal_(lin.weight); nn.init.zeros_(lin.bias)
        layers += [lin, act()]; d = width
    lin = nn.Linear(d, n_out); nn.init.xavier_normal_(lin.weight); nn.init.zeros_(lin.bias)
    layers.append(lin)
    return nn.Sequential(*layers)


def mlp_params(n_in, width, depth):
    return n_in * width + width + (depth - 1) * (width * width + width) + width + 1


def width_for_budget(n_in, depth, budget, n_nets):
    w = 4
    while n_nets * mlp_params(n_in, w + 1, depth) <= budget:
        w += 1
    return w


# method specification: every variant is a combination of these switches
#   arch: mlp | fourier | ae | gk | ts      hard: exact BC/IC (Coons x t)
#   coef: coefficient-aware rates  corners: corner product features
#   trainable: trainable rate multipliers   sigma: Fourier scale (None -> 0.1/eps)
#   gamma: two-scale exponent  width/depth/lam: overrides (published configs)
SPECS = {
    "pinn":        dict(arch="mlp"),
    "pinn_hard":   dict(arch="mlp", hard=True),
    "fourier_s1":  dict(arch="fourier", sigma=1.0),
    "fourier_s5":  dict(arch="fourier", sigma=5.0),
    "fourier_s10": dict(arch="fourier", sigma=10.0),
    "fourier_se":  dict(arch="fourier", sigma=None),
    "ae_soft":     dict(arch="ae"),
    "ae_hard":     dict(arch="ae", hard=True),
    "ae_coef":     dict(arch="ae", hard=True, coef=True),
    "ae_corner":   dict(arch="ae", hard=True, corners=True),
    "ae_cc":       dict(arch="ae", hard=True, coef=True, corners=True),
    "ae_plus":     dict(arch="ae", hard=True, coef=True, corners=True, trainable=True),
    "gkpinn":      dict(arch="gk"),
    "gkpinn_hard": dict(arch="gk", hard=True),
    "gkpinn_wang": dict(arch="gk", sub_width=100, sub_depth=2, lam=1.0),   # settings of Wang et al. (2024)
    "tsnn":        dict(arch="ts", gamma=-0.5),
    "tsnn_g1":     dict(arch="ts", gamma=-1.0),
}
# Stage 3: layer-balanced residual weighting (_b) and a 3x longer Adam budget control (_L)
for _m in ["ae_hard", "ae_coef", "ae_cc", "gkpinn_wang", "gkpinn_hard", "pinn_hard"]:
    SPECS[_m + "_b"] = dict(SPECS[_m], balance=True)
for _m in ["ae_hard", "ae_coef"]:
    SPECS[_m + "_L"] = dict(SPECS[_m], adam_mult=3)
# Tested, not reported in the paper. Hu et al. (2026), unified weighted-loss PINN: 7 one-hidden-layer sigmoid blocks (50 neurons),
#   u = u_r + [u_rx + u_sL(x/eps) + u_sR((x-1)/eps)] * [u_ry + u_sB(y/eps) + u_sT((y-1)/eps)],
#   loss weight d^2 (d = distance to the boundary) for convection problems, 1 for reaction problems,
#   soft BC/IC with unit weight. Time t is added as an input to every block (their paper is steady only).
SPECS["hu_wl"] = dict(arch="hu", lam=1.0, hu_weight=True)
# Tested, not reported in the paper. PAPINN (Cao et al. 2023): standard PINN trained by epsilon-continuation (eps = 1, 0.1, ... , target),
#   weights carried over between stages; 4 hidden layers x 20 neurons; soft BC/IC with unit weight.
SPECS["papinn"] = dict(arch="mlp", width=20, depth=4, lam=1.0, continuation=True)
# Stage 5 (a): BLC-PINN with the product form t*16x(1-x)y(1-y)*N instead of the Coons projector
#   (same features, coefficient-aware rates and balanced weight; on Ex 1/3 the coefficient-aware rates
#   equal the basic rates and on Ex 3 the weight is identically one, so this is BLC-PINN in every problem)
SPECS["blc_prod"] = dict(arch="ae", hard="prod", coef=True, balance=True)
# Stage 5 (b): competitors with 4x the training budget (4x Adam and 4x L-BFGS iterations)
for _m in ["pinn", "tsnn_g1", "fourier_s1", "gkpinn_wang"]:
    SPECS[_m + "_4x"] = dict(SPECS[_m], adam_mult=4, lbfgs_mult=4)
# Stage 6: two-scale NN with the boundary treatment of its publication for problems with boundary
#   layers (Zhuang et al., Sec. 1.3: no multiplier, w(x)=1, boundary/initial conditions by a penalty term),
#   with the authors' exponent gamma=-1/2 and with gamma=-1; penalty weight 100 as for the standard PINN
SPECS["tsnn_soft"] = dict(arch="ts", gamma=-0.5, ts_soft=True)
SPECS["tsnn_g1_soft"] = dict(arch="ts", gamma=-1.0, ts_soft=True)
for _m in ["tsnn_soft", "tsnn_g1_soft"]:
    SPECS[_m + "_4x"] = dict(SPECS[_m], adam_mult=4, lbfgs_mult=4)
TS_SOFT = ["tsnn_soft", "tsnn_g1_soft"]
# backward-compatible names (Stage 1 / Stage 2 result files)
SPECS["gkpinn_pub"] = SPECS["gkpinn_wang"]
SPECS["fourier"] = SPECS["fourier_se"]


class Model(nn.Module):
    """u(x,y,t) for every method. inputs are column tensors."""

    def __init__(self, method, prob, width=64, depth=3, n_fourier=32, seed=0):
        super().__init__()
        sp = dict(SPECS[method])
        self.m, self.p, self.sp = method, prob, sp
        self.arch = sp["arch"]
        self.hard = sp.get("hard", False)
        self.coef = sp.get("coef", False)
        self.trainable = sp.get("trainable", False)
        budget = mlp_params(3 + 4, width, depth)          # reference parameter budget
        e = prob.eps
        self.corners = []
        if self.arch == "mlp":
            self.net = mlp(3, sp.get("width", width), sp.get("depth", depth))
        elif self.arch == "hu":
            sig = nn.Sigmoid
            self.hu = nn.ModuleDict({"r": mlp(3, 50, 1, act=sig), "rx": mlp(2, 50, 1, act=sig),
                                     "ry": mlp(2, 50, 1, act=sig),
                                     "sL": mlp(2, 50, 1, act=sig), "sR": mlp(2, 50, 1, act=sig),
                                     "sB": mlp(2, 50, 1, act=sig), "sT": mlp(2, 50, 1, act=sig)})
        elif self.arch == "fourier":
            gen = torch.Generator().manual_seed(1000 + seed)
            sigma = sp["sigma"] if sp.get("sigma") else 0.1 / e
            self.register_buffer("B", torch.randn(2, n_fourier, generator=gen) * sigma)
            n_in = 3 + 2 * n_fourier
            self.net = mlp(n_in, width_for_budget(n_in, depth, budget, 1), depth)
        elif self.arch == "ae":
            if sp.get("corners"):
                self.corners = [(a, b) for a in prob.sides if a[0] == "x" for b in prob.sides if b[0] == "y"]
            if self.trainable:
                self.log_mult = nn.Parameter(torch.zeros(len(prob.sides)))
            self.net = mlp(3 + len(prob.sides) + len(self.corners), width, depth)
        elif self.arch == "gk":
            k = len(prob.sides) + 1
            if "sub_width" in sp:
                w, d = sp["sub_width"], sp["sub_depth"]
            else:
                w, d = width_for_budget(3, depth, budget, k), depth
            self.nets = nn.ModuleList([mlp(3, w, d) for _ in range(k)])
        elif self.arch == "ts":
            self.scale = e ** sp["gamma"]
            self.net = mlp(3 + 3, width, depth)
        else:
            raise ValueError(method)

    def phis(self, x, y):
        out = []
        for i, s in enumerate(self.p.sides):
            r = self.p.rate_coef(s, x, y) if self.coef else self.p.rate_orig[s]
            if self.trainable:
                r = r * torch.exp(self.log_mult[i])
            out.append(torch.exp(-r * dist(s, x, y)))
        return out

    def raw(self, x, y, t):
        tn = t / self.p.T
        a = self.arch
        if a == "mlp":
            return self.net(torch.cat([x, y, tn], 1))
        if a == "hu":
            e = self.p.eps; H = self.hu; c = lambda *v: torch.cat(v, 1)
            bx = H["rx"](c(x, tn)) + H["sL"](c(x / e, tn)) + H["sR"](c((x - 1) / e, tn))
            by = H["ry"](c(y, tn)) + H["sB"](c(y / e, tn)) + H["sT"](c((y - 1) / e, tn))
            return H["r"](c(x, y, tn)) + bx * by
        if a == "fourier":
            z = 2 * math.pi * torch.cat([x, y], 1) @ self.B
            return self.net(torch.cat([x, y, tn, torch.sin(z), torch.cos(z)], 1))
        if a == "ae":
            ph = self.phis(x, y)
            d = dict(zip(self.p.sides, ph))
            cor = [d[p] * d[q] for p, q in self.corners]
            return self.net(torch.cat([x, y, tn] + ph + cor, 1))
        if a == "gk":
            z = torch.cat([x, y, tn], 1)
            u = self.nets[0](z)
            for k, p in enumerate(self.phis(x, y)):
                u = u + self.nets[k + 1](z) * p
            return u
        if a == "ts":
            s = self.scale
            z = torch.cat([x, y, tn, s * (x - 0.5), s * (y - 0.5), torch.full_like(x, s)], 1)
            return self.net(z)

    def forward(self, x, y, t):
        if self.arch == "ts":
            if self.sp.get("ts_soft"):
                return self.raw(x, y, t)          # Stage 6: published treatment (penalty BC/IC)
            return t * x * (1 - x) * y * (1 - y) * self.raw(x, y, t)
        if not self.hard:
            return self.raw(x, y, t)
        if self.hard == "prod":
            # Stage 5: exact zero BC/IC by the simpler product form (one network evaluation);
            # 16 x(1-x) y(1-y) has maximum 1, so the network output needs no rescaling
            return t * 16 * x * (1 - x) * y * (1 - y) * self.raw(x, y, t)
        # exact zero BC via Coons projector, exact zero IC via factor t
        n = x.shape[0]
        z0, z1 = torch.zeros_like(x), torch.ones_like(x)
        X = torch.cat([x, z0, z1, x, x, z0, z1, z0, z1], 0)
        Y = torch.cat([y, y, y, z0, z1, z0, z0, z1, z1], 0)
        Tt = t.repeat(9, 1)
        N = self.raw(X, Y, Tt).split(n, 0)
        Nxy, N0y, N1y, Nx0, Nx1, N00, N10, N01, N11 = N
        P = ((1 - x) * N0y + x * N1y + (1 - y) * Nx0 + y * Nx1
             - ((1 - x) * (1 - y) * N00 + x * (1 - y) * N10 + (1 - x) * y * N01 + x * y * N11))
        return t * (Nxy - P)


# ============================================================ PDE residual
def grad(u, v):
    return torch.autograd.grad(u, v, torch.ones_like(u), create_graph=True)[0]


def operator(prob, u, x, y, t):
    """returns u_t - D Lap u + b.grad u + c u + g(u)"""
    ut, ux, uy = grad(u, t), grad(u, x), grad(u, y)
    uxx, uyy = grad(ux, x), grad(uy, y)
    r = ut - prob.D * (uxx + uyy)
    bb = prob.b(x, y)
    if bb is not None:
        r = r + bb[0] * ux + bb[1] * uy
    cc = prob.c(x, y)
    if cc is not None:
        r = r + cc * u
    gg = prob.g(u)
    if gg is not None:
        r = r + gg
    return r


def source(prob, x, y, t):
    """f at points. manufactured problems: f = operator(exact), computed in float64."""
    if not prob.has_exact:
        return prob.f_given(x, y, t).detach()
    xd, yd, td = [v.detach().double().requires_grad_(True) for v in (x, y, t)]
    ue = prob.exact(xd, yd, td)
    return operator(prob, ue, xd, yd, td).detach().to(DT)


# ============================================================ sampling
def sample_interior(prob, n, rng, layer_frac=0.5):
    nu = int(n * (1 - layer_frac))
    x = rng.random(n); y = rng.random(n); t = rng.random(n) * prob.T
    nl = n - nu
    ell = prob.layer_width()
    sides = rng.integers(0, len(prob.sides), nl)
    d = rng.exponential(ell, nl); d = np.minimum(d, 0.5)
    for k, s in enumerate(prob.sides):
        idx = nu + np.where(sides == k)[0]
        dd = d[idx - nu]
        if s == "x0": x[idx] = dd
        if s == "x1": x[idx] = 1 - dd
        if s == "y0": y[idx] = dd
        if s == "y1": y[idx] = 1 - dd
    x = np.clip(x, 1e-7, 1 - 1e-7); y = np.clip(y, 1e-7, 1 - 1e-7)
    return x, y, t


def sample_boundary(prob, n, rng):
    s = rng.random(n); t = rng.random(n) * prob.T; side = rng.integers(0, 4, n)
    x = np.where(side == 0, 0.0, np.where(side == 1, 1.0, s))
    y = np.where(side == 2, 0.0, np.where(side == 3, 1.0, s))
    return x, y, t


def T_(a):
    return torch.tensor(np.asarray(a).reshape(-1, 1), dtype=DT, device=DEV)


# ============================================================ training
def train(method, prob, seed, cfg, log=print):
    """Adam in float32, then L-BFGS in float64 (avoids premature line-search stops)."""
    global DT
    DT = torch.float32
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = Model(method, prob, width=cfg["width"], depth=cfg["depth"], seed=seed).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    pts_i = sample_interior(prob, cfg["n_f"], rng)
    pts_b = sample_boundary(prob, cfg["n_b"], rng)
    pts_0 = (rng.random(cfg["n_0"]), rng.random(cfg["n_0"]))
    soft = not (model.hard or (model.arch == "ts" and not model.sp.get("ts_soft")))
    lam = SPECS[method].get("lam", cfg["lam"])
    data = {}

    cur = {"p": prob}                        # problem used in the loss (changes during continuation)

    def build():
        xi, yi, ti = [T_(a).requires_grad_(True) for a in pts_i]
        if SPECS[method].get("balance"):
            data["wi"] = prob.balance_weight(xi, yi)
        elif SPECS[method].get("hu_weight") and prob.b(xi, yi) is not None:
            # Hu et al.: squared residual weighted by d^2 -> residual multiplied by d
            data["wi"] = torch.minimum(torch.minimum(xi, 1 - xi), torch.minimum(yi, 1 - yi)).detach()
        else:
            data["wi"] = None
        data.update(xi=xi, yi=yi, ti=ti, fi=source(cur["p"], xi, yi, ti),
                    xb=T_(pts_b[0]), yb=T_(pts_b[1]), tb=T_(pts_b[2]),
                    x0=T_(pts_0[0]), y0=T_(pts_0[1]))
        data["t0"] = torch.zeros_like(data["x0"])
    build()

    def losses():
        d = data
        u = model(d["xi"], d["yi"], d["ti"])
        res = operator(cur["p"], u, d["xi"], d["yi"], d["ti"]) - d["fi"]
        if d["wi"] is not None:
            res = res * d["wi"]
        lp = (res ** 2).mean()
        if soft:
            lb = (model(d["xb"], d["yb"], d["tb"]) ** 2).mean()
            l0 = (model(d["x0"], d["y0"], d["t0"]) ** 2).mean()
            return lp + lam * (lb + l0), lp, lb, l0
        z = torch.zeros((), device=DEV, dtype=DT)
        return lp, lp, z, z

    hist = []
    n_adam = cfg["n_adam"] * SPECS[method].get("adam_mult", 1)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    gamma = (cfg["lr_final"] / cfg["lr"]) ** (1 / max(n_adam, 1))
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    # epsilon-continuation (PAPINN): stages eps = 1, 0.1, ..., target; equal share of the Adam budget
    stages = [prob.eps]
    if SPECS[method].get("continuation"):
        stages, e_k = [], 1.0
        while e_k > prob.eps * 1.0001:
            stages.append(e_k); e_k /= 10
        stages.append(prob.eps)
    stage_len = max(n_adam // len(stages), 1)
    t_start = time.time()
    for it in range(n_adam):
        k_stage = min(it // stage_len, len(stages) - 1)
        if len(stages) > 1 and it % stage_len == 0 and it // stage_len < len(stages):
            cur["p"] = Problem(prob.name, stages[k_stage]); build()
            if k_stage == len(stages) - 1:
                best = float("inf")                  # only target-epsilon iterates count as "best"
        opt.zero_grad(set_to_none=True)
        L, lp, lb, l0 = losses()
        if not torch.isfinite(L):
            log(f"   non-finite loss at Adam step {it}; restoring best state"); break
        L.backward(); opt.step(); sch.step()
        if it % 100 == 0 or it == n_adam - 1:
            hist.append((it, L.item(), lp.item(), lb.item(), l0.item()))
            if L.item() < best and cur["p"].eps == prob.eps:
                best, best_state = L.item(), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)                 # best Adam iterate
    t_adam = time.time() - t_start

    n_lbfgs_done = 0
    if cfg["n_lbfgs"] > 0:
        if cfg.get("lbfgs_fp64", True):
            DT = torch.float64
            model.double(); build()
        pre_state = copy.deepcopy(model.state_dict())
        pre_loss = losses()[0].item()
        n_lb = cfg["n_lbfgs"] * SPECS[method].get("lbfgs_mult", 1)
        lb_opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=n_lb,
                                   max_eval=int(n_lb * 1.25), history_size=50,
                                   tolerance_grad=1e-12, tolerance_change=1e-15,
                                   line_search_fn="strong_wolfe")
        counter = {"n": 0}
        def closure():
            lb_opt.zero_grad(set_to_none=True)
            L = losses()[0]
            L.backward()
            counter["n"] += 1
            if counter["n"] % 100 == 0:
                hist.append((n_adam + counter["n"], L.item(), float("nan"), float("nan"), float("nan")))
            return L
        try:
            lb_opt.step(closure)
        except Exception as ex:
            log(f"   L-BFGS error: {ex!r}")
        n_lbfgs_done = lb_opt.state[lb_opt._params[0]].get("n_iter", 0) if lb_opt.state else 0
        post = losses()[0].item()
        if not math.isfinite(post) or post > pre_loss:
            log(f"   L-BFGS did not improve ({pre_loss:.3e} -> {post:.3e}); reverting")
            model.load_state_dict(pre_state)
    t_total = time.time() - t_start
    Lf = losses()
    info = dict(n_params=n_params, t_adam=t_adam, t_total=t_total, n_lbfgs_iter=int(n_lbfgs_done),
                final_loss=Lf[0].item(), final_pde=Lf[1].item(), final_bc=Lf[2].item(), final_ic=Lf[3].item(),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                precision_lbfgs=str(DT))
    if model.trainable:
        info["rate_multipliers"] = torch.exp(model.log_mult).tolist()
    return model, info, np.array(hist)


# ============================================================ evaluation
def eval_grid(prob, n_uniform=201, n_layer=150):
    w = min(25 * prob.layer_width(), 0.45)
    u = np.linspace(0, 1, n_uniform)
    parts = [u, 1 - np.linspace(0, w, n_layer)]
    if any(s.endswith("0") for s in prob.sides):
        parts.append(np.linspace(0, w, n_layer))
    return np.unique(np.concatenate(parts))


def predict(model, x, y, t, bs=40000):
    out = []
    for i in range(0, len(x), bs):
        with torch.no_grad():
            out.append(model(T_(x[i:i + bs]), T_(y[i:i + bs]), T_(t[i:i + bs])).cpu().numpy().ravel())
    return np.concatenate(out)


def residual_rms(model, prob, seed, n=20000, bs=4000):
    rng = np.random.default_rng(10_000 + seed)
    x, y, t = sample_interior(prob, n, rng)
    tot = 0.0
    for i in range(0, n, bs):
        xi, yi, ti = [T_(a[i:i + bs]).requires_grad_(True) for a in (x, y, t)]
        r = operator(prob, model(xi, yi, ti), xi, yi, ti) - source(prob, xi, yi, ti)
        tot += (r.detach() ** 2).sum().item()
    return math.sqrt(tot / n)


def evaluate(model, prob, seed):
    xs = eval_grid(prob)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    preds = np.array([predict(model, X.ravel(), Y.ravel(), np.full(X.size, tt)).reshape(X.shape)
                      for tt in prob.eval_times])
    ic = predict(model, X.ravel(), Y.ravel(), np.zeros(X.size))
    return xs, preds, dict(residual_rms=residual_rms(model, prob, seed), ic_max=float(np.abs(ic).max()))


# ============================================================ plans
METHODS = ["pinn", "pinn_hard", "fourier", "ae_soft", "ae_hard", "ae_plus", "gkpinn", "tsnn"]   # Stage 1
# Stage 2 (trimmed): baselines + competitors (best-faith settings) + cumulative ablation of our method
#   ablation staircase: ae_soft -> ae_hard -> ae_coef -> ae_cc -> ae_plus
#   ae_coef is identical to ae_hard on the constant-coefficient/manufactured problems -> run only where it can differ
METHODS_S2 = ["pinn", "pinn_hard", "fourier_s1", "fourier_se",
              "ae_soft", "ae_hard", "ae_coef", "ae_cc", "ae_plus",
              "gkpinn", "gkpinn_hard", "gkpinn_pub", "tsnn_g1"]
COEF_ONLY_WHERE_VARIABLE = {"ae_coef"}
PROBLEMS = ["conv_mms", "conv_var", "react_mms", "react_var"]
CFG = dict(width=64, depth=3, n_f=6000, n_b=1200, n_0=600, lam=100.0,
           lr=1e-3, lr_final=1e-4, n_adam=15000, n_lbfgs=2000)

PLAN_CFG = {
    "S1": dict(n_adam=5000, n_lbfgs=1000, lbfgs_fp64=False),
    "S2": dict(n_adam=5000, n_lbfgs=2000, lbfgs_fp64=True),
}
EPS_S2 = (1e-2, 1e-3, 1e-4)
METHODS_S3 = ["ae_hard_b", "ae_coef_b", "ae_cc_b", "gkpinn_wang_b", "pinn_hard_b",
              "ae_coef_L", "ae_hard_L", "hu_wl", "papinn"]
NEW_COMPETITORS = ["hu_wl", "papinn"]        # run on all four problems
PLAN_CFG["S3"] = dict(PLAN_CFG["S2"])
PLAN_CFG["S4"] = dict(PLAN_CFG["S2"])
PLAN_CFG["S4T"] = dict(PLAN_CFG["S2"])
PLAN_CFG["S5"] = dict(PLAN_CFG["S2"])
PLAN_CFG["S5T"] = dict(PLAN_CFG["S2"])
PLAN_CFG["S6"] = dict(PLAN_CFG["S2"])
PLAN_CFG["S6T"] = dict(PLAN_CFG["S2"])
COMP_4X = ["pinn_4x", "tsnn_g1_4x", "fourier_s1_4x", "gkpinn_wang_4x"]
BLC_FINAL = {"conv_mms": "ae_hard_b", "conv_var": "ae_coef_b", "react_mms": "ae_hard_b", "react_var": "ae_coef_b"}


def _skip_s3(m, p):
    if m.startswith("ae_coef") and p.endswith("_mms"):
        return True          # identical to ae_hard on constant-coefficient problems
    if m == "ae_hard_L" and p.endswith("_var"):
        return True          # the long-budget control uses the best variant per problem
    return False


def groups(name):
    """list of job groups; one SLURM array task runs one group"""
    if name == "S1":
        return [[(p, 1e-2, m, 0) for p in PROBLEMS for m in METHODS]]
    if name == "S2":
        return [[(p, e, m, s) for m in METHODS_S2
                 if not (m in COEF_ONLY_WHERE_VARIABLE and p.endswith("_mms"))]
                for s in (0, 1, 2) for e in EPS_S2 for p in PROBLEMS]
    if name == "S3":
        g = [[(p, e, m, s) for m in METHODS_S3 if not _skip_s3(m, p)]
             for s in (0, 1, 2) for e in EPS_S2 for p in ("conv_mms", "conv_var")]
        g += [[(p, e, m, s) for p in ("react_mms", "react_var") for m in NEW_COMPETITORS]
              for s in (0, 1, 2) for e in EPS_S2]
        return g
    if name == "S4":     # final method on Ex 4 with the balanced weight (completes the definition)
        return [[("react_var", e, "ae_coef_b", s) for e in EPS_S2] for s in (0, 1, 2)]
    if name == "S4T":    # one sequential job on ONE GPU: fair timing + solution maps for figures
        final = {"conv_mms": "ae_hard_b", "conv_var": "ae_coef_b", "react_mms": "ae_hard_b", "react_var": "ae_coef_b"}
        comp = ["gkpinn_wang", "hu_wl", "papinn", "tsnn_g1", "fourier_s1", "pinn"]
        return [[(p, 1e-4, m, 0) for p in PROBLEMS for m in [final[p]] + comp]]
    if name == "S5":
        # (a) product-form BC inside BLC-PINN: all problems, eps, seeds (12 groups x 3 runs)
        g = [[(p, e, "blc_prod", s) for e in EPS_S2] for p in PROBLEMS for s in (0, 1, 2)]
        # (b) competitors with 4x budget at eps = 1e-4 (12 groups x 4 runs)
        g += [[(p, 1e-4, m, s) for m in COMP_4X] for p in PROBLEMS for s in (0, 1, 2)]
        return g
    if name == "S5T":    # one sequential job on ONE GPU: wall time of Coons vs product form (8 runs)
        return [[(p, 1e-4, m, 0) for p in PROBLEMS for m in (BLC_FINAL[p], "blc_prod")]]
    if name == "S6":
        # two-scale NN with penalty BC/IC: all problems, eps, seeds, both exponents (12 groups x 6 runs)
        # + 4x budget at eps = 1e-4 for both exponents (same groups, 2 runs each)
        return [[(p, e, m, s) for e in EPS_S2 for m in TS_SOFT] + [(p, 1e-4, m + "_4x", s) for m in TS_SOFT]
                for p in PROBLEMS for s in (0, 1, 2)]
    if name == "S6T":    # one sequential job on ONE GPU: wall times of all compared methods + solution maps
        comp = ["gkpinn_wang", "tsnn_soft", "tsnn_g1_soft", "fourier_s1", "pinn"]
        return [[(p, 1e-4, m, 0) for p in PROBLEMS for m in [BLC_FINAL[p]] + comp]]
    if name == "A":
        return [[(p, e, m, s) for p in PROBLEMS for m in METHODS] for s in (0, 1, 2) for e in (1e-2, 1e-3)]
    raise ValueError(name)


def plan(name):
    return [j for g in groups(name) for j in g]


def run_plan(args):
    os.makedirs(args.out, exist_ok=True)
    cfg = dict(CFG)
    cfg.update(PLAN_CFG.get(args.plan, {}))
    if args.quick:
        cfg.update(n_adam=300, n_lbfgs=30)
    if args.task is not None:
        G = groups(args.plan)
        jobs = G[args.task] if args.task < len(G) else []
    else:
        jobs = plan(args.plan)
    if args.only_methods:
        jobs = [j for j in jobs if j[2] in args.only_methods]
    if args.only_problems:
        jobs = [j for j in jobs if j[0] in args.only_problems]
    if args.only_seeds is not None:
        jobs = [j for j in jobs if j[3] in args.only_seeds]
    if args.only_eps is not None:
        jobs = [j for j in jobs if j[1] in args.only_eps]
    tag_task = f"task{args.task}" if args.task is not None else "all"
    logf = open(os.path.join(args.out, f"log_{tag_task}.txt"), "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    done = sum(os.path.exists(os.path.join(args.out, f"{p}_eps{e:g}_{m}_s{s}.npz")) for p, e, m, s in jobs)
    log(f"== plan {args.plan} {tag_task}: {len(jobs)} runs, {done} already done (skipped), "
        f"{len(jobs) - done} to go | results in {args.out} | device {DEV} | cfg {cfg}")
    t_plan = time.time()
    for k, (pn, e, m, s) in enumerate(jobs):
        tag = f"{pn}_eps{e:g}_{m}_s{s}"
        fn = os.path.join(args.out, tag + ".npz")
        if os.path.exists(fn):
            continue
        prob = Problem(pn, e)
        try:
            model, info, hist = train(m, prob, s, cfg, log)
            xs, preds, ev = evaluate(model, prob, s)
            info.update(ev)
            tmp = fn[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, xs=xs, times=np.array(prob.eval_times), pred=preds.astype(np.float32),
                                hist=hist, info=json.dumps(info), cfg=json.dumps(cfg))
            os.replace(tmp, fn)                      # atomic: no half-written files after preemption
            log(f"[{k+1}/{len(jobs)}] {tag}: loss {info['final_loss']:.2e} res {info['residual_rms']:.2e} "
                f"lbfgs_it {info['n_lbfgs_iter']} params {info['n_params']} {info['t_total']/60:.1f} min "
                f"(elapsed {(time.time()-t_plan)/3600:.2f} h)")
        except Exception as ex:
            import traceback
            log(f"[{k+1}/{len(jobs)}] {tag}: FAILED {ex!r}\n{traceback.format_exc()}")


# ============================================================ self-test
def selftest():
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""), flush=True)

    # 1) manufactured source vs independent hand-derived formula (Example 1)
    e = 1e-2; p = Problem("conv_mms", e)
    a1 = -math.exp(-1 / e); a2 = -1 - a1
    xs = np.array([0.1, 0.5, 0.97, 0.995]); ys = np.array([0.3, 0.99, 0.5, 0.9]); ts = np.array([0.2, 0.7, 1.0, 0.5])
    P = lambda s: a1 + a2 * s + np.exp(-(1 - s) / e)
    P1 = lambda s: a2 + np.exp(-(1 - s) / e) / e
    P2 = lambda s: np.exp(-(1 - s) / e) / e ** 2
    sT = 1 - np.exp(-ts); u = sT * P(xs) * P(ys)
    f_hand = (np.exp(-ts) * P(xs) * P(ys) - e * sT * (P2(xs) * P(ys) + P(xs) * P2(ys))
              + sT * (P1(xs) * P(ys) + P(xs) * P1(ys)) + u * np.exp(u))
    f_ad = source(p, T_(xs), T_(ys), T_(ts)).cpu().numpy().ravel()
    rel = np.abs(f_ad - f_hand).max() / np.abs(f_hand).max()
    check("Example-1 source term (autograd vs hand formula)", rel < 1e-5, f"rel diff {rel:.1e}")

    # 2) exact solutions vanish on boundary and at t=0
    for pn in ["conv_mms", "react_mms"]:
        q = Problem(pn, 1e-3)
        xb, yb, tb = sample_boundary(q, 2000, np.random.default_rng(0))
        vb = np.abs(q.exact(xb, yb, tb, lib=np)).max()
        v0 = np.abs(q.exact(np.random.rand(100), np.random.rand(100), np.zeros(100), lib=np)).max()
        check(f"{pn}: exact solution zero on boundary / at t=0", vb < 1e-12 and v0 < 1e-12, f"{vb:.1e} {v0:.1e}")

    # 3) every method: forward, residual, backward; exact BC/IC where claimed
    rng = np.random.default_rng(1)
    for pn in PROBLEMS:
        q = Problem(pn, 1e-2)
        for m in METHODS_S2 + (METHODS_S3 if pn.startswith("conv") else []) + ["blc_prod"] + COMP_4X + TS_SOFT:
            try:
                mod = Model(m, q).to(DEV)
                xi, yi, ti = [T_(a).requires_grad_(True) for a in sample_interior(q, 64, rng)]
                r = operator(q, mod(xi, yi, ti), xi, yi, ti) - source(q, xi, yi, ti)
                (r ** 2).mean().backward()
                gfin = all(torch.isfinite(pp.grad).all() for pp in mod.parameters() if pp.grad is not None)
                msg = f"params {sum(pp.numel() for pp in mod.parameters())}"
                good = torch.isfinite(r).all().item() and gfin
                if mod.hard or (mod.arch == "ts" and not mod.sp.get("ts_soft")):
                    xb, yb, tb = sample_boundary(q, 500, rng)
                    vb = np.abs(predict(mod, xb, yb, tb)).max()
                    v0 = np.abs(predict(mod, rng.random(200), rng.random(200), np.zeros(200))).max()
                    good &= vb < 1e-5 and v0 < 1e-6
                    msg += f", max|u| on boundary {vb:.1e}, at t=0 {v0:.1e}"
                check(f"{pn:9s} {m:9s}", good, msg)
            except Exception as ex:
                check(f"{pn:9s} {m:9s}", False, repr(ex))

    # 4) timing: 50 Adam steps per method on the full point set
    print("\nTiming (seconds per 1000 Adam steps, full collocation set):", flush=True)
    q = Problem("react_var", 1e-2)
    cfg = dict(CFG, n_adam=50, n_lbfgs=0)
    for m in METHODS_S2:
        _, info, _ = train(m, q, 0, cfg, log=lambda *_: None)
        print(f"   {m:9s} {info['t_adam'] / 50 * 1000:7.1f} s", flush=True)
    print("\nL-BFGS (float64) check on ae_plus, 20 Adam + 20 L-BFGS steps:", flush=True)
    try:
        _, info, _ = train("ae_plus", Problem("conv_var", 1e-3), 0, dict(CFG, n_adam=20, n_lbfgs=20, lbfgs_fp64=True),
                           log=print)
        check("float64 L-BFGS phase runs", info["n_lbfgs_iter"] > 1 and math.isfinite(info["final_loss"]),
              f"iterations {info['n_lbfgs_iter']}, final loss {info['final_loss']:.2e}, {info['precision_lbfgs']}")
    except Exception as ex:
        check("float64 L-BFGS phase runs", False, repr(ex))
    global DT
    DT = torch.float32
    print("\nSELFTEST", "PASSED" if ok else "FAILED", flush=True)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="S2")
    ap.add_argument("--out", default="bench_results")
    ap.add_argument("--quick", action="store_true", help="tiny budgets, pipeline test")
    ap.add_argument("--only_methods", nargs="*")
    ap.add_argument("--only_problems", nargs="*")
    ap.add_argument("--only_seeds", nargs="*", type=int)
    ap.add_argument("--only_eps", nargs="*", type=float)
    ap.add_argument("--task", type=int, default=None, help="SLURM array index: run one job group")
    ap.add_argument("--n_groups", action="store_true", help="print number of job groups and exit")
    ap.add_argument("--n_runs", action="store_true", help="print total number of runs and exit")
    a = ap.parse_args()
    print("device:", DEV, "| torch", torch.__version__, flush=True)
    if a.n_groups:
        print(len(groups(a.plan))); sys.exit(0)
    if a.n_runs:
        print(len(plan(a.plan))); sys.exit(0)
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run_plan(a)
