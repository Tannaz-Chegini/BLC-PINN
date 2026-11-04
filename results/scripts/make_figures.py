"""All data figures of the paper (vector PDF). Every curve is read from the saved data.
usage: python make_figures.py [reference_dir]   (default: ../../data/references)
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "..", "data")
FIG = os.path.join(ROOT, "..", "figures")
TAB = os.path.join(ROOT, "..", "tables")
REFDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "..", "data", "references")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "serif",
                     "mathtext.fontset": "cm", "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "savefig.dpi": 600})   # resolution of rasterised parts (surfaces, error maps)
W = 5.3  # text width in inches (elsarticle preprint: 384 pt)

PROBS = ["conv_mms", "conv_var", "react_mms", "react_var"]
PTITLE = {"conv_mms": "Ex. 1 (convection)", "conv_var": "Ex. 2 (convection, variable $\\mathbf{b}$)",
          "react_mms": "Ex. 3 (reaction)", "react_var": "Ex. 4 (reaction, variable $c$)"}
EPS = [1e-2, 1e-3, 1e-4]
# method key -> (label, colour, linestyle, marker)  (Okabe-Ito based, colour-blind safe)
STY = {"blc": ("BLC-PINN", "#0072B2", "-", "o"), "gkpinn_pub": ("GKPINN", "#D55E00", "-", "s"),
       "tsnn_soft": ("Two-scale NN", "#E69F00", ":", "D"), "fourier*": ("Fourier PINN", "#999999", ":", "v"),
       "pinn": ("PINN", "#000000", "-", "^")}
M = json.load(open(os.path.join(TAB, "results_master.json")))

# ------------------------------------------------------------------ exact / reference solutions
def exact(prob, eps, X, Y, t):
    if prob == "conv_mms":
        a1 = -math.exp(-1 / eps); a2 = -1 - a1
        P = lambda s: a1 + a2 * s + np.exp(-(1 - s) / eps)
        return (1 - np.exp(-t)) * P(X) * P(Y)
    n = 1 + math.exp(-1 / eps)
    P = lambda s: 1 - (np.exp(-s / eps) + np.exp(-(1 - s) / eps)) / n
    return (1 + t - np.exp(-t)) * P(X) * P(Y)

def reference(prob, eps):
    tag = {"conv_var": "P2", "react_var": "P4"}[prob]
    r = np.load(os.path.join(REFDIR, f"ref_{tag}_eps{eps:g}_N1024.npz"))
    tf = "u_2" if prob == "conv_var" else "u_1"
    return r["x"], r["y"], r[tf]

# ------------------------------------------------------------------ Fig: the four problems (eps = 1e-2)
fig = plt.figure(figsize=(W, 4.6))
for k, p in enumerate(PROBS):
    ax = fig.add_subplot(2, 2, k + 1, projection="3d")
    if p.endswith("mms"):
        s = np.unique(np.concatenate([np.linspace(0, 1, 70), 1 - np.geomspace(1e-4, 0.15, 40), np.geomspace(1e-4, 0.15, 40)]))
        X, Y = np.meshgrid(s, s, indexing="ij"); U = exact(p, 1e-2, X, Y, 1.0)
    else:
        x, y, U = reference(p, 1e-2)
        ix = np.unique(np.linspace(0, len(x) - 1, 140).astype(int)); iy = np.unique(np.linspace(0, len(y) - 1, 140).astype(int))
        X, Y = np.meshgrid(x[ix], y[iy], indexing="ij"); U = U[np.ix_(ix, iy)]
    ax.plot_surface(X, Y, U, cmap="viridis", linewidth=0, antialiased=False, rstride=1, cstride=1, rasterized=True)
    ax.set_title(PTITLE[p] + (", $t=2$" if p == "conv_var" else ", $t=1$"), pad=0)
    ax.set_xlabel("$x$", labelpad=-6); ax.set_ylabel("$y$", labelpad=-6); ax.set_zlabel("$u$", labelpad=-6)
    ax.tick_params(pad=-3); ax.view_init(elev=26, azim=40 if p.startswith("conv") else -60)
fig.subplots_adjust(left=0.0, right=0.97, bottom=0.02, top=0.95, wspace=0.0, hspace=0.12)
fig.savefig(os.path.join(FIG, "fig_problems.pdf"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)

# ------------------------------------------------------------------ Fig: error vs eps
fig, axs = plt.subplots(2, 4, figsize=(W, 3.6), sharex=True)
for row, key, lab in [(0, "relL2", "relative $L^2$ error"), (1, "Linf", "max. error$/\\max|u|$")]:
    for c, p in enumerate(PROBS):
        ax = axs[row, c]
        for m, (l, col, ls, mk) in STY.items():
            d = M[key][p]
            med = [d[f"{e:g}"][m]["median"] for e in EPS]
            lo = [d[f"{e:g}"][m]["min"] for e in EPS]; hi = [d[f"{e:g}"][m]["max"] for e in EPS]
            ax.plot(EPS, med, ls=ls, marker=mk, color=col, lw=1.8 if m == "blc" else 1.0, ms=3.5,
                    label=l, zorder=5 if m == "blc" else 3)
            ax.fill_between(EPS, lo, hi, color=col, alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.4)
        ax.set_xticks(EPS); ax.set_ylim(2e-5 if key == "relL2" else 1e-4, 5)
        if row == 0:
            ax.set_title(PTITLE[p].replace(" (", "\n("), fontsize=8)
        else:
            ax.set_xlabel("$\\varepsilon$")
        if c > 0:
            ax.set_yticklabels([])
    axs[row, 0].set_ylabel(lab)
axs[0, 0].set_xlim(2e-2, 5e-5)      # shared x axis: eps decreases to the right
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.4)
fig.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.25, wspace=0.08, hspace=0.12)
fig.savefig(os.path.join(FIG, "fig_error_vs_eps.pdf"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)

# ------------------------------------------------------------------ figure data from the single-GPU job (eps = 1e-4)
F = dict(np.load(os.path.join(DATA, "stage4t_figdata.npz")))
# Stage 6: two-scale NN with its published (penalty) boundary treatment, from the Stage-6 single-GPU job
F.update({k: v for k, v in np.load(os.path.join(DATA, "stage6t_figdata.npz")).items() if "_tsnn_soft_" in k})
BLCRUN = {"conv_mms": "ae_hard_b", "conv_var": "ae_coef_b", "react_mms": "ae_hard_b", "react_var": "ae_coef_b"}
RUN = lambda p, m: f"{p}_eps0.0001_{m}_s0"

# ------------------------------------------------------------------ Fig: layer profiles (log distance to the outflow / right wall)
show = [("ref", "reference", "#000000", "-"), ("blc", "BLC-PINN", "#0072B2", "-"), ("gkpinn_wang", "GKPINN", "#D55E00", "--"),
        ("tsnn_soft", "Two-scale NN", "#E69F00", "-."), ("fourier_s1", "Fourier PINN", "#009E73", ":"), ("pinn", "PINN", "#999999", ":")]
# row 1: along y = 0.5 against the distance 1-x to the wall x = 1
# row 2: along the diagonal x = y against the distance to a corner: (1,1) for the convection problems
#        (outflow corner), (0,0) for the reaction problems (largest errors of both BLC-PINN and GKPINN)
CORNER = {"conv_mms": 1, "conv_var": 1, "react_mms": 0, "react_var": 0}
fig, axs = plt.subplots(2, 4, figsize=(W, 4.3))
for row in (0, 1):
    for ax, p in zip(axs[row], PROBS):
        base = RUN(p, BLCRUN[p]); xs = F[base + "_xs"].astype(float); j = int(np.argmin(abs(xs - 0.5)))
        if row == 0:
            d = 1 - xs
        else:
            d = 1 - xs if CORNER[p] == 1 else xs
        keep = d > 0
        for m, lab, col, ls in show:
            U = F[base + "_true"].astype(float) if m == "ref" else \
                F[(base if m == "blc" else RUN(p, m)) + "_u_all"].astype(float)[-1]
            u = U[:, j] if row == 0 else np.diag(U)
            if m == "blc":       # open markers, so the reference line underneath stays visible
                dk, uk = d[keep], u[keep]
                tgt = np.geomspace(dk.min(), dk.max(), 28)
                idx = np.unique([int(np.argmin(abs(np.log(dk) - np.log(v)))) for v in tgt])
                ax.plot(dk[idx], uk[idx], ls="none", marker="o", mfc="none", mec=col, mew=0.9, ms=3.6, label=lab, zorder=4)
            else:
                ax.plot(d[keep], u[keep], ls=ls, color=col, lw=1.2 if m == "ref" else 1.0, label=lab,
                        zorder=2 if m == "ref" else 3)
        ax.set_xscale("log"); ax.set_xlim(1e-5, 1); ax.grid(alpha=0.25, lw=0.4); ax.set_xticks([1e-4, 1e-2, 1])
        if row == 0:
            ax.set_title(PTITLE[p].split(" (")[0], fontsize=8); ax.set_xlabel("$1-x$  ($y=0.5$)", fontsize=7.5)
        else:
            ax.set_xlabel(("$1-x$" if CORNER[p] == 1 else "$x$") + "  ($y=x$)", fontsize=7.5)
axs[0, 0].set_ylabel("$u$ along $y=0.5$")
axs[1, 0].set_ylabel("$u$ along $y=x$")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.1, right=0.99, top=0.94, bottom=0.2, wspace=0.38, hspace=0.45)
fig.savefig(os.path.join(FIG, "fig_layer_profiles.pdf"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)

# ------------------------------------------------------------------ Fig: error maps, BLC-PINN vs GKPINN (eps = 1e-4)
# axes: distance to the walls x = 1 and y = 1 on logarithmic scales, so that the layers (width ~1e-4)
# and the interior are visible in the same picture
fig, axs = plt.subplots(2, 4, figsize=(W, 2.75))
norm = LogNorm(1e-6, 1)
for c, p in enumerate(PROBS):
    for r, (m, lab) in enumerate([("blc", "BLC-PINN"), ("gkpinn_wang", "GKPINN")]):
        run = RUN(p, BLCRUN[p]) if m == "blc" else RUN(p, m)
        xs = F[run + "_xs"].astype(float); e = np.abs(F[run + "_err"].astype(float))
        mx = np.abs(F[run + "_true"].astype(float)).max()
        keep = (xs >= 0.5) & (xs < 1)                   # half of the domain next to the walls x=1, y=1
        d = 1 - xs[keep]; E = e[np.ix_(keep, keep)] / mx
        ax = axs[r, c]
        im = ax.pcolormesh(d, d, np.clip(E.T, 1e-6, None), norm=norm, cmap="magma_r", shading="nearest", rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.5, d.min()); ax.set_ylim(0.5, d.min())
        ax.set_xticks([1e-4, 1e-2]); ax.set_yticks([1e-4, 1e-2]); ax.set_aspect("equal")
        if r == 0:
            ax.set_title(PTITLE[p].split(" (")[0], fontsize=8)
        else:
            ax.set_xlabel("$1-x$")
        if c == 0:
            ax.set_ylabel(lab + "\n$1-y$")
        else:
            ax.set_yticklabels([])
        if r == 0:
            ax.set_xticklabels([])
fig.subplots_adjust(hspace=0.08, wspace=0.08)
cb = fig.colorbar(im, ax=axs, shrink=0.95, pad=0.02); cb.set_label("$|u_{\\mathrm{NN}}-u|/\\max|u|$")
fig.savefig(os.path.join(FIG, "fig_error_maps.pdf"), bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig: BLC-PINN solution of Ex. 2 at three times (eps = 1e-4)
run = RUN("conv_var", "ae_coef_b"); xs = F[run + "_xs"].astype(float); U = F[run + "_u_all"].astype(float)
T = F[run + "_times"]
rel = [r for r in json.load(open(os.path.join(DATA, "stage4t_metrics.json")))
       if r["prob"] == "conv_var" and r["method"] == "ae_coef_b"][0]["relL2_all_times"]
fig, axs = plt.subplots(1, 3, figsize=(W, 2.75))
for k, ax in enumerate(axs):                  # own colour scale per panel (the solution grows with t)
    im = ax.contourf(xs, xs, U[k].T, levels=24, cmap="viridis")
    try:
        im.set_edgecolor("face")      # removes hairline gaps between contour bands in PDF viewers
    except Exception:
        [c.set_edgecolor("face") for c in im.collections]
    ax.set_aspect("equal"); ax.set_title(f"$t={T[k]:g}$\nrel. $L^2$ error {rel[k]:.1e}", fontsize=7.5)
    ax.set_xlabel("$x$"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.9, pad=0.22, aspect=18); cb.ax.tick_params(labelsize=6); cb.locator = matplotlib.ticker.MaxNLocator(4); cb.update_ticks()
axs[0].set_ylabel("$y$")
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIG, "fig_time_ex2.pdf"), bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig: BLC-PINN training history (eps = 1e-4)
fig, ax = plt.subplots(figsize=(3.1, 2.1))
for p, col in zip(PROBS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]):
    h = F[RUN(p, BLCRUN[p]) + "_hist"]
    ax.semilogy(h[:, 0], h[:, 1], color=col, lw=1.0, label=PTITLE[p].split(" (")[0])
ax.axvline(5000, color="#888888", lw=0.6, ls="--"); ax.text(5100, ax.get_ylim()[1] * 0.3, "L-BFGS", fontsize=7, color="#555555")
ax.text(200, ax.get_ylim()[1] * 0.3, "Adam", fontsize=7, color="#555555")
ax.set_xlabel("iteration"); ax.set_ylabel("training loss"); ax.grid(alpha=0.25, lw=0.4); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_training.pdf")); plt.close(fig)
print("figures:", sorted(os.listdir(FIG)))
