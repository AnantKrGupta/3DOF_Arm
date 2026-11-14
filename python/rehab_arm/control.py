"""Joint-space tracking controllers, a batched simulator and a real-coded GA.

Both controllers share the same structure (fair comparison):

    tau = G_hat(q_m) + Kp e + Ki z + Kd e_dot,      e = q_ref - q_m,  z = int e dt

* PID : Kp, Ki, Kd diagonal  -> 9 gains tuned by a genetic algorithm
* LQR : Kp, Ki, Kd full 3x3  -> from LQI (integral-augmented LQR) on the
        gravity-compensated model linearised at the mean task inertia M0.

The plant is the full nonlinear Lagrangian model (optionally with
parameter mismatch, sensor noise and torque disturbances); control runs
at 1 kHz with zero-order hold and torque saturation + conditional
integration anti-windup.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_continuous_are

from .dynamics import ArmDynamics
from .trajectory import WaypointTrajectory


@dataclass
class Gains:
    Kp: np.ndarray  # (N,3,3)
    Ki: np.ndarray
    Kd: np.ndarray

    @staticmethod
    def from_pid_vector(x):
        """x (N,9) = [Kp1 Kp2 Kp3 Ki1 Ki2 Ki3 Kd1 Kd2 Kd3]."""
        x = np.atleast_2d(x)
        diag = lambda v: np.einsum("ni,ij->nij", v, np.eye(3))  # noqa: E731
        return Gains(diag(x[:, 0:3]), diag(x[:, 3:6]), diag(x[:, 6:9]))

    @staticmethod
    def from_matrices(Kp, Ki, Kd):
        return Gains(Kp[None], Ki[None], Kd[None])


@dataclass
class Scenario:
    name: str = "nominal"
    payload_true: float | None = None      # actual payload (model uses config value)
    mass_scale: float = 1.0                # scales link masses of the TRUE plant
    noise_q: float = 2e-4                  # rad   (encoder noise std)
    noise_qd: float = 5e-3                 # rad/s (velocity estimate noise std)
    disturbance: list = field(default_factory=list)  # [(t0, t1, joint, Nm)]
    seed: int = 0


def lqi_gains(model: ArmDynamics, M0: np.ndarray, e_max, ev_max, z_max, tau_fb_max, dt: float | None = 1e-3):
    """Integral-augmented LQR on M0 xdd + B xd = tau_fb, Bryson's rule weights.
    If dt is given the design is done on the ZOH-discretised model (DARE),
    which is what the 1 kHz controller actually sees."""
    n = 3
    Minv = np.linalg.inv(M0)
    A = np.zeros((3 * n, 3 * n))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 3:6] = -Minv @ np.diag(model.B)
    A[6:9, 0:3] = np.eye(3)
    Bm = np.zeros((3 * n, n))
    Bm[3:6] = Minv
    Q = np.diag(np.concatenate([1 / np.asarray(e_max) ** 2, 1 / np.asarray(ev_max) ** 2,
                                1 / np.asarray(z_max) ** 2]))
    R = np.diag(1 / np.asarray(tau_fb_max) ** 2)
    if dt:
        from scipy.linalg import expm, solve_discrete_are
        Mexp = expm(np.block([[A, Bm], [np.zeros((n, 4 * n))]]) * dt)
        Ad, Bd = Mexp[:3 * n, :3 * n], Mexp[:3 * n, 3 * n:]
        Qd, Rd = Q * dt, R * dt
        P = solve_discrete_are(Ad, Bd, Qd, Rd)
        K = np.linalg.solve(Rd + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        cl_poles = np.log(np.linalg.eigvals(Ad - Bd @ K).astype(complex)) / dt
    else:
        P = solve_continuous_are(A, Bm, Q, R)
        K = np.linalg.solve(R, Bm.T @ P)      # u = -K x, x = [q-qr, qd-qrd, int(q-qr)]
        cl_poles = np.linalg.eigvals(A - Bm @ K)
    # tau_fb = Kq (qr-q) + Kv (qrd-qd) + Kz int(qr-q)
    return K[:, 0:3], K[:, 6:9], K[:, 3:6], K, cl_poles


def mean_task_inertia(model: ArmDynamics, traj: WaypointTrajectory, n=200):
    t = np.linspace(0, traj.duration, n)
    q, _, _ = traj(t)
    return model.M(q).mean(axis=0)


def simulate(gains: Gains, scenario: Scenario | None = None, traj: WaypointTrajectory | None = None,
             T: float | None = None, dt: float = 1e-3, model: ArmDynamics | None = None, record=True):
    """Simulate N controllers in parallel. Returns dict of arrays (K, N, 3)."""
    sc = scenario or Scenario()
    traj = traj or WaypointTrajectory()
    model = model or ArmDynamics()
    params = model.params
    plant = ArmDynamics(params, payload=sc.payload_true if sc.payload_true is not None else params.mp)
    plant.P[4] *= sc.mass_scale
    plant.P[5] *= sc.mass_scale
    rng = np.random.default_rng(sc.seed)
    T = T or traj.duration
    K = int(round(T / dt)) + 1
    t = np.arange(K) * dt
    qr, qrd, _ = traj(t)
    N = gains.Kp.shape[0]
    q = np.repeat(qr[:1], N, 0).copy()
    qd = np.zeros((N, 3))
    z = np.zeros((N, 3))
    tau_max = params.tau_max
    diverged = np.zeros(N, bool)
    out = {k: np.zeros((K, N, 3)) for k in ("q", "qd", "tau", "e")} if record else {}
    ise = np.zeros(N); effort = np.zeros(N); dtau2 = np.zeros(N); sat = np.zeros(N)
    tau_prev = model.G(q)

    def f(qq, vv, tau):
        return vv, plant.forward(qq, vv, tau)

    for k in range(K):
        qm = q + rng.normal(0, sc.noise_q, q.shape)
        qdm = qd + rng.normal(0, sc.noise_qd, qd.shape)
        e = qr[k] - qm
        ed = qrd[k] - qdm
        tau = (model.G(qm) + np.einsum("nij,nj->ni", gains.Kp, e)
               + np.einsum("nij,nj->ni", gains.Ki, z) + np.einsum("nij,nj->ni", gains.Kd, ed))
        tau_c = np.clip(tau, -tau_max, tau_max)
        saturated = np.abs(tau) > tau_max
        z = z + np.where(saturated, 0.0, e) * dt          # conditional-integration anti-windup
        tau_applied = tau_c.copy()
        for (t0, t1, j, val) in sc.disturbance:
            if t0 <= t[k] < t1:
                tau_applied[:, j] += val
        # RK4 plant step with ZOH torque
        k1q, k1v = f(q, qd, tau_applied)
        k2q, k2v = f(q + dt / 2 * k1q, qd + dt / 2 * k1v, tau_applied)
        k3q, k3v = f(q + dt / 2 * k2q, qd + dt / 2 * k2v, tau_applied)
        k4q, k4v = f(q + dt * k3q, qd + dt * k3v, tau_applied)
        etrue = qr[k] - q
        ise += np.sum(etrue**2, 1) * dt
        effort += np.sum(tau_c**2, 1) * dt
        dtau2 += np.sum(((tau_c - tau_prev) / dt) ** 2, 1) * dt
        sat += np.any(saturated, 1) * dt
        tau_prev = tau_c
        if record:
            out["q"][k], out["qd"][k], out["tau"][k], out["e"][k] = q, qd, tau_c, etrue
        q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        qd = qd + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        bad = ~np.isfinite(q).all(1) | (np.abs(q) > 10).any(1) | (np.abs(qd) > 100).any(1)
        if bad.any():
            diverged |= bad
            q[bad], qd[bad], z[bad] = qr[k], 0.0, 0.0
    out.update(t=t, qr=qr, qrd=qrd, ise=ise, effort=effort, dtau2=dtau2, sat_time=sat, diverged=diverged)
    return out


def pid_cost(res, w_ise=1e4, w_effort=1e-3, w_dtau=2e-6, w_sat=50.0):
    """Scalar GA fitness: tracking ISE + torque effort + torque-rate (noise
    amplification / chatter) + saturation time; diverged runs get 1e6."""
    J = (w_ise * res["ise"] + w_effort * res["effort"] + w_dtau * res["dtau2"] + w_sat * res["sat_time"])
    return np.where(res["diverged"], 1e6, J)


def metrics(res, idx=0):
    e = res["e"][:, idx] * 180 / np.pi
    tau = res["tau"][:, idx]
    dt = res["t"][1] - res["t"][0]
    return {
        "rmse_deg": np.sqrt(np.mean(e**2, 0)).round(3).tolist(),
        "max_err_deg": np.abs(e).max(0).round(3).tolist(),
        "rms_tau_Nm": np.sqrt(np.mean(tau**2, 0)).round(3).tolist(),
        "peak_tau_Nm": np.abs(tau).max(0).round(2).tolist(),
        "tau_rate_rms_Nm_s": np.sqrt(np.mean((np.diff(tau, axis=0) / dt) ** 2, 0)).round(1).tolist(),
        "saturation_time_s": float(res["sat_time"][idx]),
        "ISE_rad2s": float(res["ise"][idx]),
        "diverged": bool(res["diverged"][idx]),
    }


# ----------------------------------------------------------------------- GA
def genetic_algorithm(fitness, lb, ub, pop=40, gens=30, elite=2, pc=0.9, pm=0.15,
                      tournament=3, blx_alpha=0.3, seed=1, log=print):
    """Real-coded GA: tournament selection, BLX-alpha crossover, Gaussian
    mutation (sigma = 10% of range, decaying), elitism. `fitness` maps
    (pop, dim) -> (pop,) and is evaluated batch-wise."""
    rng = np.random.default_rng(seed)
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    dim = lb.size
    X = lb + (ub - lb) * rng.random((pop, dim))
    F = fitness(X)
    history = []
    for g in range(gens):
        order = np.argsort(F)
        X, F = X[order], F[order]
        history.append((g, float(F[0]), float(np.median(F))))
        log(f"  gen {g:2d}  best J = {F[0]:.4f}  median J = {np.median(F):.4f}")
        children = [X[i].copy() for i in range(elite)]
        sigma = 0.1 * (ub - lb) * (1 - g / gens) + 1e-3 * (ub - lb)
        while len(children) < pop:
            def pick():
                c = rng.integers(0, pop, tournament)
                return X[c[np.argmin(F[c])]]
            a, b = pick(), pick()
            if rng.random() < pc:
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                d = hi - lo
                child = rng.uniform(lo - blx_alpha * d, hi + blx_alpha * d)
            else:
                child = a.copy()
            m = rng.random(dim) < pm
            child[m] += rng.normal(0, sigma[m])
            children.append(np.clip(child, lb, ub))
        Xn = np.array(children)
        Fn = np.empty(pop)
        Fn[:elite] = F[:elite]
        Fn[elite:] = fitness(Xn[elite:])
        X, F = Xn, Fn
    order = np.argsort(F)
    history.append((gens, float(F[order[0]]), float(np.median(F))))
    return X[order[0]], float(F[order[0]]), np.array(history)
