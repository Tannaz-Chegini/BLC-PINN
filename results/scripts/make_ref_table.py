"""Accuracy of the finite-difference reference solver (Appendix table).
(a) convergence against the exact solutions of Ex. 1 and Ex. 3;
(b) self-convergence N=512 vs N=1024 for the references of Ex. 2 and Ex. 4 (the ones used for grading)."""
import os, sys, time, json, numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference_solver"))
from fd_solver2 import shishkin, solve
REF = sys.argv[1]                              # folder with ref_P*_N*.npz
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
rows_a = []
CACHE = os.path.join(OUT, "reference_accuracy.json")
if os.path.exists(CACHE) and "--recompute" not in sys.argv:
    rows_a = [tuple(r) for r in json.load(open(CACHE))["exact"]]
for prob in ([] if rows_a else ["Ex1", "Ex3"]):
    for eps in [1e-2, 1e-3, 1e-4]:
        errs = []
        for N in [128, 256, 512]:
            nt = 2 * N if N < 512 else 800
            if prob == "Ex1":
                a1 = -np.exp(-1 / eps); a2 = -1 - a1
                P = lambda s: a1 + a2 * s + np.exp(-(1 - s) / eps); P1 = lambda s: a2 + np.exp(-(1 - s) / eps) / eps
                P2 = lambda s: np.exp(-(1 - s) / eps) / eps ** 2
                ue = lambda X, Y, t: (1 - np.exp(-t)) * P(X) * P(Y)
                def f(X, Y, t):
                    s = 1 - np.exp(-t); u = s * P(X) * P(Y)
                    return np.exp(-t)*P(X)*P(Y) - eps*s*(P2(X)*P(Y)+P(X)*P2(Y)) + s*(P1(X)*P(Y)+P(X)*P1(Y)) + u*np.exp(u)
                x = shishkin(N, 2.5 * eps * np.log(N), (False, True))
                S = solve(eps, lambda X, Y: (1 + 0 * X, 1 + 0 * Y), lambda X, Y: 0 * X, lambda u: u * np.exp(u), f, 1.0, x, x, nt, (1.0,))
            else:
                n = 1 + np.exp(-1 / eps); E = lambda s: np.exp(-s / eps) + np.exp(-(1 - s) / eps)
                Ps = lambda s: 1 - E(s) / n; Pss = lambda s: -E(s) / (n * eps ** 2)
                tau = lambda t: 1 + t - np.exp(-t); taup = lambda t: 1 + np.exp(-t); g = lambda u: (u - 4) / (5 - u)
                ue = lambda X, Y, t: tau(t) * Ps(X) * Ps(Y)
                def f(X, Y, t):
                    u = tau(t) * Ps(X) * Ps(Y)
                    return taup(t)*Ps(X)*Ps(Y) - eps**2*tau(t)*(Pss(X)*Ps(Y)+Ps(X)*Pss(Y)) + g(u)
                x = shishkin(N, 2.5 * eps * np.log(N), (True, True))
                S = solve(eps ** 2, lambda X, Y: (0 * X, 0 * Y), lambda X, Y: 0 * X, g, f, 1.0, x, x, nt, (1.0,))
            X, Y = np.meshgrid(x, x, indexing="ij"); U = ue(X, Y, 1.0)
            errs.append(np.abs(S[1.0] - U).max() / np.abs(U).max())
            print(prob, eps, N, errs[-1], flush=True)
        rows_a.append((prob, eps, errs))
# (b) Ex. 2 and Ex. 4 (no exact solution).
#  Ex. 2, eps <= 1e-3: the scheme is first order in the coarse region (upwinding), so the grading reference is the
#  Richardson extrapolation 2 u_1024 - u_512; its error is estimated by comparing with 2 u_512 - u_256
#  (eps = 1e-3 uses consistent upwinding in the whole coarse region, files upc_*; see fd_solver_upc.py).
#  Ex. 2, eps = 1e-2 and Ex. 4: central differences (second order); reference u_1024, error estimated by |u_512 - u_1024|.
def trap(x):
    w = np.zeros_like(x); d = np.diff(x); w[:-1] += d / 2; w[1:] += d / 2; return w
def load(tag, eps, N, pre="ref"):
    for f in (f"{pre}_{tag}_eps{eps:g}_N{N}.npz", f"{pre}_{tag}_eps{eps:g}.npz"):
        if os.path.exists(os.path.join(REF, f)) and (N == 512 or f.endswith(f"N{N}.npz")):
            return np.load(os.path.join(REF, f))
    raise FileNotFoundError((tag, eps, N, pre))
rows_b = []
for tag, ex, tf in [("P2", "Ex.~2", "u_2"), ("P4", "Ex.~4", "u_1")]:
    for eps in [1e-2, 1e-3, 1e-4]:
        rich = tag == "P2" and eps <= 1e-3
        pre = "upc" if (tag == "P2" and eps == 1e-3) else "ref"
        Ns = (256, 512, 1024) if rich else (512, 1024)
        S = {N: load(tag, eps, N, pre) for N in Ns}
        g = S[512]; X, Y = np.meshgrid(g["x"], g["y"], indexing="ij"); pts = np.stack([X.ravel(), Y.ravel()], 1)
        U = {N: RGI((S[N]["x"], S[N]["y"]), S[N][tf])(pts).reshape(X.shape) for N in Ns}
        W = np.outer(trap(g["x"]), trap(g["y"]))
        nrm = lambda a: np.sqrt((W * a ** 2).sum() / (W * U[1024] ** 2).sum())
        mx = lambda a: np.abs(a).max() / np.abs(U[1024]).max()
        if rich:
            p = np.log2(nrm(U[256] - U[512]) / nrm(U[512] - U[1024]))
            d = (2 * U[1024] - U[512]) - (2 * U[512] - U[256])
            rows_b.append((ex, eps, "extrapolated", float(p), float(nrm(d)), float(mx(d))))
        else:
            d = U[512] - U[1024]
            rows_b.append((ex, eps, "$N=1024$", None, float(nrm(d)), float(mx(d))))
        print(rows_b[-1], flush=True)
def sci(x):
    e = int(np.floor(np.log10(x))); m = x / 10 ** e
    return f"{m:.1f}e$-${-e}"
EPS = {1e-2: "$10^{-2}$", 1e-3: "$10^{-3}$", 1e-4: "$10^{-4}$"}
L = [r"\begin{table}[H]", r"\centering\small",
     r"\caption{Accuracy of the finite-difference reference solver. Top: maximum nodal error relative to "
     r"$\max|u|$ at $t=T$ against the exact solutions of Ex.~1 and Ex.~3 ($N$ intervals per direction). "
     r"Bottom: references used for grading Ex.~2 and Ex.~4 and their estimated errors at $t=T$ (relative $L^2$ "
     r"and maximum relative to $\max|u|$). `Extrapolated': Richardson extrapolation $2u_{1024}-u_{512}$, with the "
     r"observed convergence rate from $N=256,512,1024$ in brackets and the error estimated by the difference from "
     r"$2u_{512}-u_{256}$; otherwise the $N=1024$ solution, with the error estimated by $|u_{512}-u_{1024}|$.}",
     r"\label{tab:reference}", r"\begin{tabular}{llccc}", r"\toprule",
     r"Problem & $\varepsilon$ & $N=128$ & $N=256$ & $N=512$ \\", r"\midrule"]
for i, (p, e, errs) in enumerate(rows_a):
    L.append(((("Ex.~1" if p == "Ex1" else "Ex.~3")) if e == 1e-2 else "") + f" & {EPS[e]} & " + " & ".join(sci(v) for v in errs) + r" \\")
L += [r"\midrule", r"Problem & $\varepsilon$ & reference (rate) & \multicolumn{2}{c}{estimated error: rel.\ $L^2$ / max} \\", r"\midrule"]
for ex, e, kind, p, l2, m in rows_b:
    ref = kind + (f" ({p:.2f})" if p is not None else "")
    L.append((ex if e == 1e-2 else "") + f" & {EPS[e]} & {ref} & \\multicolumn{{2}}{{c}}{{{sci(l2)} / {sci(m)}}}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_reference.tex"), "w").write("\n".join(L) + "\n")
json.dump(dict(exact=rows_a, refs=rows_b), open(os.path.join(OUT, "reference_accuracy.json"), "w"), indent=1)
print("written")
