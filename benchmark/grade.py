"""Errors of every saved run against the exact solution (Ex. 1, Ex. 3) or the reference solution (Ex. 2, Ex. 4).
usage: python grade.py <results_dir> refs_on_eval_grid_v2.npz <name> [allmaps]   -> <name>_metrics.json, <name>_table.txt"""
import sys, os, glob, json, math
import numpy as np

res_dir, ref_path = sys.argv[1], sys.argv[2]
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "stage2"
ALLMAPS = len(sys.argv) > 4 and sys.argv[4] == "allmaps"
REF = np.load(ref_path)


def exact(prob, eps, X, Y, t):
    if prob == "conv_mms":
        a1 = -math.exp(-1 / eps); a2 = -1 - a1
        P = lambda s: a1 + a2 * s + np.exp(-(1 - s) / eps)
        return (1 - np.exp(-t)) * P(X) * P(Y)
    if prob == "react_mms":
        n = 1 + math.exp(-1 / eps)
        P = lambda s: 1 - (np.exp(-s / eps) + np.exp(-(1 - s) / eps)) / n
        return (1 + t - np.exp(-t)) * P(X) * P(Y)


def trap(xs):
    w = np.zeros_like(xs); d = np.diff(xs); w[:-1] += d / 2; w[1:] += d / 2; return w


def layer_mask(prob, eps, X, Y):
    ell = 2 * eps if prob == "conv_var" else eps
    if prob.startswith("react"):
        return np.minimum.reduce([X, 1 - X, Y, 1 - Y]) < 10 * ell
    return (1 - X < 10 * ell) | (1 - Y < 10 * ell)


rows, problems = [], []
files = sorted(f for f in glob.glob(os.path.join(res_dir, "*.npz")) if ".tmp." not in f)
for f in files:
    base = os.path.basename(f)[:-4]
    try:
        prob, rest = base.split("_eps")
        epstag = rest.split("_")[0]; eps = float(epstag)
        method, seed = rest[len(epstag) + 1:].rsplit("_s", 1); seed = int(seed)
        d = np.load(f, allow_pickle=True)
        info = json.loads(str(d["info"]))
        xs, times, pred = d["xs"].astype(float), d["times"], d["pred"].astype(float)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        W = np.outer(trap(xs), trap(xs)); L = layer_mask(prob, eps, X, Y)
        per_t = []
        for k, t in enumerate(times):
            if prob.endswith("mms"):
                true = exact(prob, eps, X, Y, float(t))
            else:
                key = f"{prob}_eps{eps:g}"
                assert np.allclose(REF[key + "_xs"], xs, atol=1e-12), "grid mismatch"
                true = REF[key + "_u"][k].astype(float)
            e = pred[k] - true; mx = np.abs(true).max()
            bnd = np.concatenate([pred[k][0], pred[k][-1], pred[k][:, 0], pred[k][:, -1]])
            per_t.append(dict(t=float(t),
                              relL2=math.sqrt((W * e ** 2).sum() / (W * true ** 2).sum()),
                              Linf=float(np.abs(e).max() / mx),
                              Linf_layer=float(np.abs(e[L]).max() / mx),
                              Linf_out=float(np.abs(e[~L]).max() / mx) if (~L).any() else None,
                              bc=float(np.abs(bnd).max()), umin=float(pred[k].min())))
        r = dict(prob=prob, eps=eps, method=method, seed=seed, **per_t[-1],
                 relL2_all_times=[p["relL2"] for p in per_t],
                 residual=info.get("residual_rms"), ic=info.get("ic_max"),
                 minutes=info["t_total"] / 60, minutes_adam=info["t_adam"] / 60,
                 params=info["n_params"], lbfgs=info["n_lbfgs_iter"],
                 final_loss=info["final_loss"], device=info.get("device"),
                 rate_mult=info.get("rate_multipliers"))
        rows.append(r)
    except Exception as ex:
        rows.append(dict(file=base, error=repr(ex)))

json.dump(rows, open(f"{PREFIX}_metrics.json", "w"))
ok = [r for r in rows if "error" not in r]
bad = [r for r in rows if "error" in r]
lines = [f"graded {len(ok)} runs, {len(bad)} errors"]
for p in ["conv_mms", "conv_var", "react_mms", "react_var"]:
    for e in (1e-2, 1e-3, 1e-4):
        R = [r for r in ok if r["prob"] == p and r["eps"] == e]
        if not R:
            continue
        lines.append(f"\n## {p}  eps={e:g}   relL2 median over seeds (min-max) | Linf median | minutes")
        for m in sorted({r["method"] for r in R}, key=lambda m: np.median([r["relL2"] for r in R if r["method"] == m])):
            v = [r["relL2"] for r in R if r["method"] == m]
            li = [r["Linf"] for r in R if r["method"] == m]
            mi = [r["minutes"] for r in R if r["method"] == m]
            lines.append(f"  {m:12s} {np.median(v):.2e} ({min(v):.1e}-{max(v):.1e})  {np.median(li):.2e}  {np.median(mi):5.1f}  n={len(v)}")
open(f"{PREFIX}_table.txt", "w").write("\n".join(lines))
print("\n".join(lines[:1]))
print(f"wrote {PREFIX}_metrics.json and {PREFIX}_table.txt")

# ---- solution maps for paper figures
#   default : seed 0, eps = 1e-3, final time, 5 methods (Stage 2)
#   allmaps : every run in the folder; all saved times (float16, for plotting),
#             final-time error map (float16; computed here against the exact / reference solution),
#             and the training-loss history
fig = {}
for f in files:
    base = os.path.basename(f)[:-4]
    if ALLMAPS or "_eps0.001_" in base and base.endswith("_s0") and \
       any(base.endswith(f"_{m}_s0") for m in ["pinn", "ae_soft", "ae_hard", "ae_plus", "gkpinn_pub"]):
        d = np.load(f, allow_pickle=True)
        fig[base + "_xs"] = d["xs"]
        if not ALLMAPS:
            fig[base + "_u"] = d["pred"][-1].astype(np.float32)
            continue
        prob, rest = base.split("_eps"); eps = float(rest.split("_")[0])
        xs = d["xs"].astype(float); X, Y = np.meshgrid(xs, xs, indexing="ij")
        t_end = float(d["times"][-1])
        true = exact(prob, eps, X, Y, t_end) if prob.endswith("mms") else REF[f"{prob}_eps{eps:g}_u"][-1].astype(float)
        fig[base + "_times"] = d["times"]
        fig[base + "_u_all"] = d["pred"].astype(np.float16)
        fig[base + "_err"] = (d["pred"][-1].astype(float) - true).astype(np.float16)
        fig[base + "_true"] = true.astype(np.float16)
        fig[base + "_hist"] = d["hist"]
if fig:
    np.savez_compressed(f"{PREFIX}_figdata.npz", **fig)
    print(f"wrote {PREFIX}_figdata.npz ({sum(k.endswith('_xs') for k in fig)} runs)")
