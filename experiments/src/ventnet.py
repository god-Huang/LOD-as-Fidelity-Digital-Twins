"""
ventnet.py — Mine ventilation network solver (Atkinson law + node-pressure Newton method).

Governing equations (steady, incompressible, fully turbulent — the standard mine ventilation
network assumptions, Atkinson 1850 / McPherson 1993):

    branch law        dp_i = R_i * Q_i * |Q_i|                        [Pa]
    Atkinson          R_i  = integral( alpha * P(s) / A(s)**3 ) ds    [N s^2 m^-8]
    node continuity   sum_{i incident} (+/-) Q_i + q_ext,j = 0
    fan branch        dp_i = (R_i + r_f |Q_i|) Q_i |Q_i| - H0

Constant air density is assumed, so elevation enters only through the piezometric pressure
datum; natural ventilation driven by density contrast is deliberately out of scope and is
listed as a limitation.

Solution method. Substituting the branch law into continuity gives a nonlinear system in the
nodal piezometric pressures. It is solved by a damped Newton iteration with backtracking on
the residual norm, warm-started from the *linearised* network (constant conductance 1/R_i),
which makes the iteration contractive from the first step. Near-zero-flow branches are
smoothed by Q_EPS in the derivative so the Jacobian stays nonsingular; verification
V5 shows the smoothing term is inactive at the operating points used here, and V4 shows the solution is
invariant to branch subdivision.

The quadratic parallel reduction used by the verification is
    R_parallel = ( sum_i R_i^(-1/2) )^(-2),
NOT the linear harmonic rule, because dp = R Q^2 is nonlinear in Q.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

RHO_AIR = 1.20            # kg/m^3, dry air near standard conditions
G = 9.81                  # m/s^2
Q_EPS = 1.0e-3            # m^3 s^-1 : smoothing flow in the Newton derivative, bounded cap
ALPHA_MAIN = 0.0045       # N s^2 m^-4, supported main roadway (typical metal-mine value)
ALPHA_RAMP = 0.0055       # N s^2 m^-4, unpaved curved ramp
ALPHA_TURN = 0.0075       # N s^2 m^-4, small unsupported branch / crosscut
CLASS_ALPHA = {"主巷道": ALPHA_MAIN, "斜坡道": ALPHA_RAMP, "岔道": ALPHA_TURN}


class VentNet:
    """Node-pressure ventilation network with quadratic (Atkinson) branch laws."""

    def __init__(self, n_nodes, a, b, R, q_ext, ground=0,
                 kind=None, fan_h0=None, fan_r=None):
        self.n = int(n_nodes)
        self.a = np.asarray(a, int); self.b = np.asarray(b, int)
        self.R = np.maximum(np.asarray(R, float), 1e-9)
        self.q_ext = np.asarray(q_ext, float)
        self.ground = int(ground)
        m = len(self.a)
        self.kind = np.zeros(m, int) if kind is None else np.asarray(kind, int)
        self.fan_h0 = np.zeros(m) if fan_h0 is None else np.asarray(fan_h0, float)
        self.fan_r = np.zeros(m) if fan_r is None else np.asarray(fan_r, float)
        self._free = np.ones(self.n, bool); self._free[self.ground] = False
        self._fi = np.where(self._free)[0]
        self._Qlag = None                 # lagged branch flow for the fan self-resistance

    # ------------------------------------------------------------------ #
    def _effective_R(self, dp):
        Reff = self.R.copy()
        f = self.kind == 1
        if f.any():
            q0 = np.abs(self._Qlag[f]) if self._Qlag is not None else \
                np.sqrt(np.abs(dp[f]) / self.R[f])
            Reff[f] = np.maximum(self.R[f] + self.fan_r[f] * q0, 1e-9)
        return Reff

    def flows(self, p):
        """Branch flows and the exact Newton derivative dQ/d(dp).

        dp = R Q|Q|  =>  Q = sign(dp) sqrt(|dp|/R),  dQ/d(dp) = 1 / (2 R |Q|).
        |Q| is smoothed by Q_EPS so the derivative stays finite for a closing-to-zero
        branch while remaining exact (relative error < 1e-4) for |Q| > 10 m^3/s.
        """
        dp = p[self.a] - p[self.b] - self.fan_h0
        Reff = self._effective_R(dp)
        sq = np.sqrt(np.abs(dp) / Reff)
        Q = np.sign(dp) * sq
        dQdp = 1.0 / (2.0 * Reff * (sq + Q_EPS))
        return Q, dQdp

    def residual(self, p):
        Q, g = self.flows(p)
        r = np.zeros(self.n)
        np.add.at(r, self.a, -Q)
        np.add.at(r, self.b, Q)
        return r + self.q_ext, Q, g

    def jacobian(self, g):
        rows = np.r_[self.a, self.a, self.b, self.b]
        cols = np.r_[self.a, self.b, self.a, self.b]
        vals = np.r_[-g, g, g, -g]
        return sparse.coo_matrix((vals, (rows, cols)), shape=(self.n, self.n)).tocsr()

    def _linear_start(self):
        """Warm start: the topological distribution of the network, i.e. the linear network
        obtained by freezing each quadratic branch at dp = 1 Pa, so G_i = R_i^(-1/2)."""
        g = 1.0 / np.sqrt(self.R)
        J = self.jacobian(g)[self._fi][:, self._fi]
        r = -self.residual(np.zeros(self.n))[0][self._free]
        eps = 1e-9 * max(1.0, float(np.abs(J.diagonal()).max()))
        J = (J + sparse.eye(J.shape[0]) * eps).tocsc()
        try:
            p = spsolve(J, r)
        except Exception:
            p = np.linalg.lstsq(J.toarray(), r, rcond=None)[0]
        p0 = np.zeros(self.n); p0[self._free] = np.nan_to_num(np.asarray(p).ravel(), nan=0.0)
        return p0

    def solve(self, tol=1e-8, maxit=100, p0=None, verbose=False, warm=True):
        scale = max(1.0, float(np.abs(self.q_ext).max()))
        if p0 is not None:
            p = np.asarray(p0, float).copy()
        elif warm:
            p = self._linear_start()
        else:
            p = np.zeros(self.n)
        rn_prev = np.inf
        it = 0
        for it in range(maxit):
            res, Q, g = self.residual(p)
            rn = float(np.abs(res[self._free]).max())
            if verbose and it % 10 == 0:
                print(f"   it{it:3d}  ||r||inf={rn:.3e}")
            if rn < tol * scale:
                break
            J = self.jacobian(g)[self._fi][:, self._fi]
            eps = 1e-12 * max(1.0, float(np.abs(J.diagonal()).max()))
            J = (J + sparse.eye(J.shape[0]) * eps).tocsc()
            try:
                step = spsolve(J, -res[self._free])
            except Exception:
                step = np.linalg.lstsq(J.toarray(), -res[self._free], rcond=None)[0]
            step = np.nan_to_num(np.asarray(step).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            cap = 0.4 * np.maximum(np.abs(p[self._free]), 20.0)
            step = np.clip(step, -cap, cap)
            self._Qlag = Q.copy()
            lam, base = 1.0, rn
            for _ in range(30):
                pt = p.copy(); pt[self._free] += lam * step
                rt = float(np.abs(self.residual(pt)[0][self._free]).max())
                if rt <= base or lam < 1e-5:
                    break
                lam *= 0.5
            p[self._free] += lam * step
            if lam < 1e-5 and rn >= rn_prev:
                break
            rn_prev = rn
        res, Q, g = self.residual(p)
        rn = float(np.abs(res[self._free]).max())
        dp_br = p[self.a] - p[self.b] - self.fan_h0
        return dict(p=p, Q=Q, dp_branch=dp_br, iters=it + 1,
                    mass_residual=rn, converged=bool(rn < tol * scale),
                    residual_rel=rn / scale)

    # ------------------------------------------------------------------ #
    def equivalent_resistance(self, sol=None):
        """Total mine resistance Re = |dp(src -> snk)| / Q_total^2."""
        sol = sol or self.solve()
        src = int(np.argmax(self.q_ext)); snk = int(np.argmin(self.q_ext))
        Qt = float(abs(self.q_ext[src]))
        dp = float(sol["p"][snk] - sol["p"][src])
        return abs(dp) / Qt ** 2, dp, Qt

    def air_power(self, sol=None):
        """Total aerodynamic power dissipated in the network, W."""
        sol = sol or self.solve()
        return float(np.sum(self.R * np.abs(sol["Q"]) ** 3))


# --------------------------------------------------------------------------- #
def r_series(*Rs):
    return float(np.sum(Rs))


def r_parallel(*Rs):
    """Quadratic-law parallel reduction: (sum R^-1/2)^-2."""
    Rs = np.asarray(Rs, float)
    return float((np.sum(Rs ** -0.5)) ** -2)


def integrate_resistance(alpha, s, A, P):
    """R = integral alpha*P/A^3 ds along a branch (trapezoid over stations)."""
    s = np.asarray(s, float); A = np.asarray(A, float); P = np.asarray(P, float)
    ok = np.isfinite(s) & np.isfinite(A) & np.isfinite(P) & (A > 0.3) & (P > 0.3)
    if ok.sum() < 2:
        return np.nan
    return float(np.trapezoid(alpha * P[ok] / A[ok] ** 3, s[ok]))


def design_resistance(alpha, L, A0, P0):
    """Resistance of the same branch under the conventional fixed design section."""
    return float(alpha * L * P0 / A0 ** 3)
