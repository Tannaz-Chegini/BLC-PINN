"""Builds every results table of the paper from the graded metrics files.
No number in the manuscript's tables is typed by hand.

Inputs : stage2_metrics.json, stage3_metrics.json, stage4_metrics.json, stage6_metrics.json, stage6t_metrics.json
Outputs: ../tables/*.tex, ../tables/results_master.json
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "..", "data")
OUT = os.path.join(ROOT, "..", "tables")
os.makedirs(OUT, exist_ok=True)

def load(name):
    return [r for r in json.load(open(os.path.join(DATA, name))) if "error" not in r]

S2, S3, S4, S4T = (load(f) for f in ["stage2_metrics.json", "stage3_metrics.json",
                                     "stage4_metrics.json", "stage4t_metrics.json"])
S6, S6T = load("stage6_metrics.json"), load("stage6t_metrics.json")
# Stage 6: two-scale NN with the boundary treatment of its publication (penalty BC/IC, gamma = -1/2);
# it replaces the Stage-2 runs 'tsnn_g1' (product-form BC). S6T (single-GPU job) gives the wall times.
RUNS = S2 + S3 + S4 + S6     # timing jobs (S4T, S6T) are used only for cost and figures

PROBS = ["conv_mms", "conv_var", "react_mms", "react_var"]
PNAME = {"conv_mms": "Ex.~1", "conv_var": "Ex.~2", "react_mms": "Ex.~3", "react_var": "Ex.~4"}
EPS = [1e-2, 1e-3, 1e-4]
EPSTEX = {1e-2: "$10^{-2}$", 1e-3: "$10^{-3}$", 1e-4: "$10^{-4}$"}

# ---------------------------------------------------------------- method definitions
# BLC-PINN = layer coordinates (coefficient-aware rates) + exact BC/IC + layer-balanced weight.
# On Ex. 1 and Ex. 3 the coefficient-aware rates equal the basic rates (constant coefficients /
# manufactured), and on Ex. 3 the balanced weight is identically 1 (A_i = eps^2 * (1/eps)^2 = 1),
# so the Stage-2 run 'ae_hard' is the same algorithm there.
BLC = {"conv_mms": "ae_hard_b", "conv_var": "ae_coef_b", "react_mms": "ae_hard", "react_var": "ae_coef_b"}
COMPETITORS = [("gkpinn_pub", "GKPINN"), ("tsnn_soft", "Two-scale NN"), ("fourier*", "Fourier PINN"), ("pinn", "PINN")]
# two-line column headers for the wide tables
HEAD = {"BLC-PINN": ("BLC-PINN", ""), "GKPINN": ("GKPINN", ""), "Two-scale NN": ("Two-scale NN", ""),
        "Fourier PINN": ("Fourier PINN", ""), "PINN": ("PINN", "")}
HIGHLIGHT = set()     # columns to mark with \chg{} (used only for a tracked-changes version of the manuscript)
# Fourier PINN: the better of the two scales (sigma = 1, sigma = 0.1/eps) per problem and eps (by median)

def runs(p, e, m):
    return [r for r in RUNS if r["prob"] == p and r["eps"] == e and r["method"] == m]

def resolve(p, e, m):
    if m == "blc":
        return BLC[p]
    if m == "fourier*":
        c = [mm for mm in ("fourier_s1", "fourier_se") if runs(p, e, mm)]
        return min(c, key=lambda mm: np.median([r["relL2"] for r in runs(p, e, mm)]))
    return m

def stat(p, e, m, key="relL2"):
    mm = resolve(p, e, m)
    v = [r[key] for r in runs(p, e, mm)]
    assert len(v) == 3, (p, e, mm, len(v))
    return float(np.median(v)), float(min(v)), float(max(v)), mm

def sci(x, digits=1):
    if not np.isfinite(x):
        return "--"
    ex = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** ex
    if round(m, digits) >= 10:
        m /= 10; ex += 1
    return f"{m:.{digits}f}e{'$-$' if ex < 0 else '$+$'}{abs(ex)}"

master = {}

# ---------------------------------------------------------------- main tables (L2 and Linf)
def main_table(key, fname, caption, label):
    cols = [("blc", "BLC-PINN")] + COMPETITORS
    lines = [r"\begin{table}[!htb]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
             "Problem & $\\varepsilon$ & " + " & ".join(HEAD[c[1]][0] for c in cols) + r" \\", r"\midrule"]
    for p in PROBS:
        for i, e in enumerate(EPS):
            vals = [stat(p, e, m, key) for m, _ in cols]
            best = int(np.argmin([v[0] for v in vals]))
            cells = []
            for j, v in enumerate(vals):
                s = sci(v[0])
                s = r"\textbf{" + s + "}" if j == best else s
                if cols[j][0] in HIGHLIGHT:
                    s = r"\chg{" + s + "}"        # blue only in the tracked version (\chg is a no-op otherwise)
                cells.append(s)
                master.setdefault(key, {}).setdefault(p, {}).setdefault(f"{e:g}", {})[cols[j][0]] = \
                    dict(median=v[0], min=v[1], max=v[2], run_name=v[3])
            lines.append((PNAME[p] if i == 0 else "") + " & " + EPSTEX[e] + " & " + " & ".join(cells) + r" \\")
        if p != PROBS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")

main_table("relL2", "tab_main_l2.tex",
           r"Relative $L^2$ error at the final time, against the exact solution (Ex.~1, Ex.~3) or the reference solution (Ex.~2, Ex.~4); median of three seeds, best value in bold. "
           r"Fourier PINN: frequency scale $\sigma=1$, the better of the two scales tried in every case.", "tab:main-l2")
main_table("Linf", "tab_main_linf.tex",
           r"Maximum pointwise error at the final time, relative to $\max|u|$ (median of three seeds; "
           r"best value in bold). Values near or above 1 mean that the boundary layer is not resolved.",
           "tab:main-linf")

# ---------------------------------------------------------------- ablation
ABL = {
    "conv_mms": [("ae_soft", "features with rate $1/\\varepsilon$, penalty BC"), ("ae_hard", "+ exact BC/IC"),
                 ("ae_hard_L", "+ exact BC/IC, $3\\times$ longer training (control)"),
                 ("ae_hard_b", "+ exact BC/IC + balanced weight (BLC-PINN)")],
    "conv_var": [("ae_soft", "features with rate $1/\\varepsilon$, penalty BC"), ("ae_hard", "+ exact BC/IC"),
                 ("ae_coef", "+ coefficient-aware rates"),
                 ("ae_coef_L", "+ coefficient-aware rates, $3\\times$ longer training (control)"),
                 ("ae_coef_b", "+ balanced weight (BLC-PINN)")],
    "react_mms": [("ae_soft", "features with rate $1/\\varepsilon$, penalty BC"), ("ae_hard", "+ exact BC/IC (= BLC-PINN)")],
    "react_var": [("ae_soft", "features with rate $1/\\varepsilon$, penalty BC"), ("ae_hard", "+ exact BC/IC"),
                  ("ae_coef", "+ coefficient-aware rates"), ("ae_coef_b", "+ balanced weight (BLC-PINN)")],
}
lines = [r"\begin{table}[!htb]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Ablation study: each component added one at a time (relative $L^2$ error, median of three seeds). "
         r"On Ex.~1 and Ex.~3 the coefficient-aware rates coincide with the basic rates; on Ex.~3 the "
         r"balanced weight is identically one. The rows marked `control' use three times as many Adam "
         r"iterations and no weight. The first row of each problem uses the original exponential features "
         r"with rate $1/\varepsilon$ and penalty boundary conditions.}", r"\label{tab:ablation}",
         r"\begin{tabular}{l>{\raggedright\arraybackslash}p{0.42\textwidth}ccc}", r"\toprule",
         r"Problem & Variant & $\varepsilon=10^{-2}$ & $\varepsilon=10^{-3}$ & $\varepsilon=10^{-4}$ \\", r"\midrule"]
for p in PROBS:
    for i, (m, lab) in enumerate(ABL[p]):
        v = [stat(p, e, m)[0] for e in EPS]
        master.setdefault("ablation", {}).setdefault(p, {})[m] = v
        lines.append((PNAME[p] if i == 0 else "") + " & " + lab + " & " + " & ".join(sci(x) for x in v) + r" \\")
    if p != PROBS[-1]:
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_ablation.tex"), "w").write("\n".join(lines) + "\n")

# design choices that did not help (text numbers, saved for the manuscript)
extra = {}
for p, base, cor, tr in [("conv_mms", "ae_hard_b", "ae_cc_b", None), ("conv_var", "ae_coef_b", "ae_cc_b", None),
                         ("conv_mms", "ae_hard", "ae_cc", "ae_plus"), ("conv_var", "ae_coef", "ae_cc", "ae_plus"),
                         ("react_mms", "ae_hard", "ae_cc", "ae_plus"), ("react_var", "ae_coef", "ae_cc", "ae_plus")]:
    for m in [base, cor, tr]:
        if m:
            extra.setdefault(p, {})[m] = [stat(p, e, m)[0] for e in EPS]
for p in ["conv_mms", "conv_var"]:
    for m in ["pinn_hard", "pinn_hard_b", "gkpinn_wang_b"]:
        extra.setdefault(p, {})[m] = [stat(p, e, m)[0] for e in EPS]
        extra[p][m + "_Linf"] = [stat(p, e, m, "Linf")[0] for e in EPS]
master["extra"] = extra

# ---------------------------------------------------------------- cost (single GPU type, A40)
names = [("blc", "BLC-PINN")] + COMPETITORS
MAC = {"blc": "BLC", "gkpinn_pub": "GK", "tsnn_soft": "TS", "fourier*": "FO", "pinn": "PI"}
mac = ["% generated by make_tables.py: wall time (min, one A40, eps=1e-4, seed 0, mean of 4 problems) and parameters"]
cost = {}
for m, lab in names:
    rr = []
    for p in PROBS:
        mm = BLC[p] if m == "blc" else ("fourier_s1" if m == "fourier*" else m)
        mm = {"gkpinn_pub": "gkpinn_wang"}.get(mm, mm)  # same configuration, renamed in Stage 4
        if mm == "ae_hard" and p == "react_mms":
            mm = "ae_hard_b"     # timing job ran the (identical) weighted variant
        hit = [r for r in S6T if r["prob"] == p and r["method"] == mm]
        assert len(hit) == 1 and hit[0]["device"] == "NVIDIA A40", (p, mm)
        rr.append(hit[0])
    mins = float(np.mean([r["minutes"] for r in rr]))
    params = sorted({r["params"] for r in rr})
    ptxt = f"{params[0]:,}" if len(params) == 1 else f"{params[0]:,}--\\allowbreak{{}}{params[-1]:,}"
    cost[m] = dict(minutes=mins, params=params)
    mac.append(rf"\newcommand{{\time{MAC[m]}}}{{{mins:.1f}}}\newcommand{{\par{MAC[m]}}}{{{ptxt}}}")
mac.append(rf"\newcommand{{\timeratio}}{{{cost['blc']['minutes']/cost['pinn']['minutes']:.1f}}}")
open(os.path.join(OUT, "cost_macros.tex"), "w").write("\n".join(mac) + "\n")
master["cost"] = cost

json.dump(master, open(os.path.join(OUT, "results_master.json"), "w"), indent=1)
print("tables written:", sorted(f for f in os.listdir(OUT) if f.endswith(".tex")))
