"""
Reference solver v2 (fast) for
    u_t - D*Lap(u) + b(x,y).grad(u) + c(x,y) u + g(u) = f(x,y,t)  on (0,1)^2 x (0,T],
    u = 0 on the boundary, u(.,0) = 0.

* Space: tensor Shishkin mesh, refined at x=0 and/or x=1 (same for y).
  Second-order central differences; at nodes where the mesh Peclet number
  |b| h / (2D) exceeds 1 (coarse region only) the convection term switches to
  first-order upwinding (hybrid scheme, Stynes & Roos), which keeps the
  scheme oscillation-free for small D.
* Time: IMEX-BDF2 (SBDF2): linear part implicit, nonlinear g explicit with
  second-order extrapolation. The matrix is constant -> one sparse LU, then
  cheap solves.  First step: IMEX backward Euler.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def shishkin(N, width, sides):
    """Piecewise-uniform mesh with N intervals; `width` = layer width estimate;
    sides = (left_layer, right_layer)."""
    L, R = sides
    tau = min(0.25 if (L and R) else 0.5, width)
    if L and R:
        n = N // 4
        a = np.linspace(0, tau, n + 1)
        m = np.linspace(tau, 1 - tau, N - 2 * n + 1)[1:]
        c = np.linspace(1 - tau, 1, n + 1)[1:]
        return np.concatenate([a, m, c])
    if R:
        a = np.linspace(0, 1 - tau, N // 2 + 1)
        c = np.linspace(1 - tau, 1, N // 2 + 1)[1:]
        return np.concatenate([a, c])
    if L:
        a = np.linspace(0, tau, N // 2 + 1)
        c = np.linspace(tau, 1, N // 2 + 1)[1:]
        return np.concatenate([a, c])
    return np.linspace(0, 1, N + 1)


def _axis_stencils(x):
    hm = x[1:-1] - x[:-2]
    hp = x[2:] - x[1:-1]
    d2 = (2 / (hm * (hm + hp)), -2 / (hm * hp), 2 / (hp * (hm + hp)))
    d1c = (-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp)))
    return hm, hp, d2, d1c


def assemble(x, y, D, b, c):
    nx, ny = len(x) - 2, len(y) - 2
    X, Y = np.meshgrid(x[1:-1], y[1:-1], indexing="ij")
    b1, b2 = b(X, Y)
    cc = c(X, Y)
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []

    def add(r, cl, v, mask):
        rows.append(r[mask]); cols.append(cl[mask]); vals.append(v[mask])

    for axis, (grid, bb) in enumerate([(x, b1), (y, b2)]):
        hm, hp, d2, d1c = _axis_stencils(grid)
        shape = (nx, ny)
        if axis == 0:
            HM, HP = np.broadcast_to(hm[:, None], shape), np.broadcast_to(hp[:, None], shape)
            D2 = [np.broadcast_to(s[:, None], shape) for s in d2]
            D1 = [np.broadcast_to(s[:, None], shape) for s in d1c]
        else:
            HM, HP = np.broadcast_to(hm[None, :], shape), np.broadcast_to(hp[None, :], shape)
            D2 = [np.broadcast_to(s[None, :], shape) for s in d2]
            D1 = [np.broadcast_to(s[None, :], shape) for s in d1c]
        pe = np.abs(bb) * np.maximum(HM, HP) / (2 * D)
        up = np.maximum(HM, HP) > 0.5 * np.diff(grid).max()   # consistent: upwind in the whole coarse region, central in the fine region
        # upwind first derivative
        lo_u = np.where(bb > 0, -1 / HM, 0.0)
        di_u = np.where(bb > 0, 1 / HM, -1 / HP)
        hi_u = np.where(bb > 0, 0.0, 1 / HP)
        lo1 = np.where(up, lo_u, D1[0]); di1 = np.where(up, di_u, D1[1]); hi1 = np.where(up, hi_u, D1[2])
        lo = -D * D2[0] + bb * lo1
        di = -D * D2[1] + bb * di1
        hi = -D * D2[2] + bb * hi1
        I = idx
        if axis == 0:
            nb_lo = np.full(shape, -1); nb_lo[1:, :] = idx[:-1, :]
            nb_hi = np.full(shape, -1); nb_hi[:-1, :] = idx[1:, :]
        else:
            nb_lo = np.full(shape, -1); nb_lo[:, 1:] = idx[:, :-1]
            nb_hi = np.full(shape, -1); nb_hi[:, :-1] = idx[:, 1:]
        full = np.ones(shape, bool)
        add(I.ravel(), I.ravel(), di.ravel(), full.ravel())
        add(I.ravel(), nb_lo.ravel(), lo.ravel(), (nb_lo >= 0).ravel())
        add(I.ravel(), nb_hi.ravel(), hi.ravel(), (nb_hi >= 0).ravel())
    add(idx.ravel(), idx.ravel(), cc.ravel(), np.ones(nx * ny, bool))
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * ny, nx * ny)).tocsc()
    return A, X, Y


def solve(D, b, c, g, f, T, x, y, nt, store_times=()):
    A, X, Y = assemble(x, y, D, b, c)
    n = A.shape[0]
    I = sp.identity(n, format="csc")
    dt = T / nt
    lu1 = spla.splu((I / dt + A).tocsc())
    lu2 = spla.splu((1.5 * I / dt + A).tocsc())
    u0 = np.zeros(n)
    u1 = lu1.solve(u0 / dt + f(X, Y, dt).ravel() - g(u0))
    out = {}
    def rec(t, u):
        for ts in store_times:
            if abs(t - ts) < 0.5 * dt:
                out[ts] = u.copy()
    rec(dt, u1)
    um, uc = u0, u1
    for k in range(2, nt + 1):
        t = k * dt
        rhs = (2 * uc - 0.5 * um) / dt + f(X, Y, t).ravel() - g(2 * uc - um)
        un = lu2.solve(rhs)
        um, uc = uc, un
        rec(t, uc)
    def full(u):
        F = np.zeros((len(x), len(y))); F[1:-1, 1:-1] = u.reshape(len(x) - 2, len(y) - 2); return F
    return {k: full(v) for k, v in out.items()}
