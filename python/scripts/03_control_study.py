"""Step 3 (Python mirror of the Simulink study) - GA-tuned PID vs LQR (LQI).

Usage:  python python/scripts/03_control_study.py [--quick]
Outputs -> results/control_python/
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rehab_arm import ROOT, WaypointTrajectory  # noqa: E402
from rehab_arm.dynamics import ArmDynamics  # noqa: E402
from rehab_arm.control import (Gains, Scenario, genetic_algorithm, lqi_gains, mean_task_inertia,  # noqa: E402
                               metrics, pid_cost, simulate)

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="small GA for a smoke test")
args = ap.parse_args()

OUT = ROOT / "results" / "control_python"
OUT.mkdir(parents=True, exist_ok=True)
DEG = np.pi / 180
model = ArmDynamics()
traj = WaypointTrajectory()
P = model.params
summary = {}

# ------------------------------------------------------------------ LQR (LQI)
M0 = mean_task_inertia(model, traj)
lqr_w = dict(e_max=[2 * DEG] * 3, ev_max=[20 * DEG] * 3, z_max=[0.5 * DEG] * 3, tau_fb_max=0.5 * P.tau_max)
Kp_l, Ki_l, Kd_l, K, poles = lqi_gains(model, M0, **lqr_w, dt=1e-3)
summary["lqr"] = {"M0": M0.round(5).tolist(), "Kp": Kp_l.round(2).tolist(), "Ki": Ki_l.round(2).tolist(),
                  "Kd": Kd_l.round(3).tolist(),
                  "closed_loop_poles_equiv_rad_s": np.round(poles, 2).astype(str).tolist(),
                  "weights": {k: np.round(np.asarray(v), 4).tolist() for k, v in lqr_w.items()}}
print("LQR Kp diag", np.diag(Kp_l).round(1), "Ki", np.diag(Ki_l).round(1), "Kd", np.diag(Kd_l).round(2))

# ------------------------------------------------------------------ GA-tuned PID
lb = np.array([50, 50, 25, 20, 20, 10, 8, 8, 4], dtype=float)
ub = np.array([800, 800, 400, 800, 800, 400, 50, 50, 25], dtype=float)
tune_sc = Scenario(name="tuning", seed=123)


def fitness(X):
    res = simulate(Gains.from_pid_vector(X), tune_sc, traj, model=model, record=False)
    return pid_cost(res)


pop, gens = (12, 4) if args.quick else (40, 30)
t0 = time.time()
x_best, J_best, hist = genetic_algorithm(fitness, lb, ub, pop=pop, gens=gens, seed=7)
summary["ga"] = {"pop": pop, "generations": gens, "best_cost": J_best, "wall_time_s": round(time.time() - t0, 1),
                 "Kp": x_best[0:3].round(2).tolist(), "Ki": x_best[3:6].round(2).tolist(),
                 "Kd": x_best[6:9].round(3).tolist(), "bounds_lb": lb.tolist(), "bounds_ub": ub.tolist()}
print("GA best", x_best.round(2), "J", J_best)
lqr_cost = float(pid_cost(simulate(Gains.from_matrices(Kp_l, Ki_l, Kd_l), tune_sc, traj, model=model, record=False))[0])
summary["lqr"]["cost_same_metric"] = lqr_cost

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(hist[:, 0], hist[:, 1], "o-", label="best")
ax.semilogy(hist[:, 0], hist[:, 2], "s--", label="median")
ax.axhline(lqr_cost, color="r", ls=":", label="LQR (same cost)")
ax.set_xlabel("generation"); ax.set_ylabel("cost J"); ax.set_title("GA convergence (PID gains)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "ga_convergence.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ evaluation
gains = Gains(np.concatenate([Gains.from_pid_vector(x_best).Kp, Kp_l[None]]),
              np.concatenate([Gains.from_pid_vector(x_best).Ki, Ki_l[None]]),
              np.concatenate([Gains.from_pid_vector(x_best).Kd, Kd_l[None]]))
names = ["GA-PID", "LQR"]
scenarios = [
    Scenario("nominal", seed=1),
    Scenario("heavy_payload_1.5kg", payload_true=1.5, seed=2),
    Scenario("mass+20%_noise_x3", mass_scale=1.2, noise_q=6e-4, noise_qd=1.5e-2, seed=3),
    Scenario("spasm_disturbance", disturbance=[(3.6, 3.9, 2, 4.0), (7.5, 7.8, 1, -6.0)], seed=4),
]
summary["scenarios"] = {}
results = {}
for sc in scenarios:
    r = simulate(gains, sc, traj, model=model)
    results[sc.name] = r
    summary["scenarios"][sc.name] = {n: metrics(r, i) for i, n in enumerate(names)}
    print(sc.name, {n: summary["scenarios"][sc.name][n]["rmse_deg"] for n in names})

r = results["nominal"]
t = r["t"]
fig, axs = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
for j in range(3):
    axs[0, j].plot(t, r["qr"][:, j] / DEG, "k", lw=2, label="reference")
    for i, (n, c) in enumerate(zip(names, ["tab:blue", "tab:orange"])):
        axs[0, j].plot(t, r["q"][:, i, j] / DEG, "--", color=c, label=n)
        axs[1, j].plot(t, r["e"][:, i, j] / DEG, color=c, label=n)
        axs[2, j].plot(t, r["tau"][:, i, j], color=c, lw=0.8, label=n)
    axs[0, j].set_title(P.joint_names[j]); axs[0, j].set_ylabel("q [deg]")
    axs[1, j].set_ylabel("error [deg]"); axs[2, j].set_ylabel("tau [Nm]"); axs[2, j].set_xlabel("t [s]")
axs[0, 0].legend(); axs[1, 0].legend()
fig.suptitle("Nominal rehab-task tracking: GA-tuned PID vs LQR (LQI)")
fig.tight_layout(); fig.savefig(OUT / "tracking_nominal.png", dpi=150); plt.close(fig)

r = results["spasm_disturbance"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for i, n in enumerate(names):
    axs[0].plot(t, r["e"][:, i, 2] / DEG, label=n)
    axs[1].plot(t, r["e"][:, i, 1] / DEG, label=n)
for ax, ttl in zip(axs, ["elbow error (4 Nm pulse @3.6 s)", "shoulder-flexion error (-6 Nm pulse @7.5 s)"]):
    ax.set_title(ttl); ax.set_xlabel("t [s]"); ax.set_ylabel("deg"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "disturbance_rejection.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
xs = np.arange(len(scenarios))
for i, n in enumerate(names):
    v = [np.mean(summary["scenarios"][s.name][n]["rmse_deg"]) for s in scenarios]
    ax.bar(xs + (i - 0.5) * 0.35, v, 0.35, label=n)
ax.set_xticks(xs); ax.set_xticklabels([s.name for s in scenarios]); ax.set_ylabel("mean joint RMSE [deg]")
ax.set_title("Robustness comparison"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "robustness_rmse.png", dpi=150); plt.close(fig)

lines = ["| scenario | controller | RMSE [deg] (j1,j2,j3) | max err [deg] | peak tau [Nm] | tau-rate RMS [Nm/s] |",
         "|---|---|---|---|---|---|"]
for s in scenarios:
    for n in names:
        m = summary["scenarios"][s.name][n]
        lines.append(f"| {s.name} | {n} | {m['rmse_deg']} | {max(m['max_err_deg'])} | {m['peak_tau_Nm']} | {m['tau_rate_rms_Nm_s']} |")
(OUT / "metrics_table.md").write_text("\n".join(lines) + "\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("\n".join(lines))
