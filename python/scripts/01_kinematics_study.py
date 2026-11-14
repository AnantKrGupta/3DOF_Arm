"""Step 1 - DH kinematics, workspace and singularity analysis.

Outputs -> results/kinematics/*.png + kinematics_summary.json
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rehab_arm import ThreeDofArm, WaypointTrajectory, ROOT  # noqa: E402

OUT = ROOT / "results" / "kinematics"
OUT.mkdir(parents=True, exist_ok=True)
arm = ThreeDofArm()
P = arm.p
DEG = np.pi / 180
rng = np.random.default_rng(42)
summary = {}


def draw_arm(ax, q, color="k", lw=3, label=None):
    k = arm.keypoints(q)
    ax.plot(k[:, 0], k[:, 1], k[:, 2], "-o", color=color, lw=lw, ms=4, label=label)


# ------------------------------------------------------------- 1. DH table
dh_md = ["| i | a_i [m] | alpha_i | d_i | theta_i | joint |", "|---|---|---|---|---|---|"]
for i, (row, name) in enumerate(zip(arm.dh_table(np.zeros(3)), P.joint_names), 1):
    dh_md.append(f"| {i} | {row[0]:.3f} | {row[1] / DEG:+.0f} deg | {row[2]:.0f} | q{i} | {name} |")
(OUT / "dh_table.md").write_text("\n".join(dh_md) + "\n")
print("\n".join(dh_md))

# ------------------------------------------------------------- 2. workspace
N = 200_000
Q = arm.sample_joint_space(N, rng)
X = arm.fk_batch(Q)
w = arm.manipulability(Q)
hull = ConvexHull(X)
vox = 0.02
occupied = np.unique(np.floor(X / vox).astype(int), axis=0)
summary["workspace"] = {
    "samples": N,
    "max_reach_m": float(np.max(np.linalg.norm(X - P.shoulder_pos, axis=1))),
    "convex_hull_volume_m3": float(hull.volume),
    "voxel_volume_m3_2cm": float(len(occupied) * vox**3),
    "x_range_m": [float(X[:, 0].min()), float(X[:, 0].max())],
    "y_range_m": [float(X[:, 1].min()), float(X[:, 1].max())],
    "z_range_m": [float(X[:, 2].min()), float(X[:, 2].max())],
    "manipulability_mean": float(w.mean()), "manipulability_max": float(w.max()),
    "fraction_near_singular_(w<0.1*max)": float(np.mean(w < 0.1 * w.max())),
}

sub = rng.choice(N, 20000, replace=False)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(*X[sub].T, c=w[sub], s=1, cmap="viridis")
for q, c in [([0.17, 0.17, 0.35], "r"), ([0.26, 1.05, 0.52], "m"), ([1.3, 0.35, 0.7], "orange")]:
    draw_arm(ax, q, c)
ax.set_xlabel("X fwd [m]"); ax.set_ylabel("Y left [m]"); ax.set_zlabel("Z up [m]")
ax.set_title("Reachable workspace (joint limits), colour = manipulability")
fig.colorbar(sc, shrink=0.6, label="w = |det Jv| [m^3]")
ax.set_box_aspect((np.ptp(X[:, 0]), np.ptp(X[:, 1]), np.ptp(X[:, 2])))
fig.tight_layout(); fig.savefig(OUT / "workspace_3d.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, (i, j, t) in zip(axs, [(0, 2, "Sagittal (X-Z)"), (1, 2, "Frontal (Y-Z)"), (0, 1, "Transverse (X-Y)")]):
    hb = ax.hexbin(X[:, i], X[:, j], C=w, reduce_C_function=np.max, gridsize=70, cmap="viridis")
    ax.plot(P.shoulder_pos[i], P.shoulder_pos[j], "r*", ms=14, label="shoulder")
    ax.set_aspect("equal"); ax.set_title(t); ax.set_xlabel("XYZ"[i] + " [m]"); ax.set_ylabel("XYZ"[j] + " [m]")
    ax.legend(loc="lower right")
fig.colorbar(hb, ax=axs, shrink=0.8, label="max manipulability in bin")
fig.savefig(OUT / "workspace_projections.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------- 3. singularity map
q2 = np.linspace(*P.limits[1], 300)
q3 = np.linspace(-0.2, np.pi, 300)
G2, G3 = np.meshgrid(q2, q3)
D = -P.L1 * P.L2 * np.sin(G3) * (P.L1 * np.cos(G2) + P.L2 * np.cos(G2 + G3))
traj = WaypointTrajectory()
tt = np.linspace(0, traj.duration, 600)
qt, _, _ = traj(tt)
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(G2 / DEG, G3 / DEG, np.abs(D), 30, cmap="magma")
ax.contour(G2 / DEG, G3 / DEG, D, levels=[0], colors="cyan", linewidths=2)
ax.axhspan(P.limits[2, 0] / DEG, P.limits[2, 1] / DEG, color="w", alpha=0.08)
ax.plot(qt[:, 1] / DEG, qt[:, 2] / DEG, "w--", lw=2, label="rehab task trajectory")
ax.plot([], [], "c-", label="det(Jv)=0 (singular loci)")
ax.set_xlabel("q2 shoulder flexion [deg]"); ax.set_ylabel("q3 elbow flexion [deg]")
ax.set_title("|det Jv| (independent of q1); singular when s3=0 or L1c2+L2c23=0")
fig.colorbar(cf, label="|det Jv| [m^3]"); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "singularity_map.png", dpi=150); plt.close(fig)

# verify numerically that J loses rank exactly on the analytic loci
checks = []
sing_qs = [np.array([0.3, 0.6, 0.0]),                       # elbow straight
           np.array([0.5, np.arctan2(P.L1 + P.L2 * np.cos(1.0), P.L2 * np.sin(1.0)), 1.0])]  # hand on q1 axis
for q in sing_qs:
    s = np.linalg.svd(arm.jacobian_pos(q), compute_uv=False)
    checks.append({"q_deg": (q / DEG).round(2).tolist(), "singular_values": s.round(6).tolist()})
summary["singularity_checks"] = checks

# ------------------------------------------------------------- 4. singular configurations
fig = plt.figure(figsize=(12, 5))
for k, (q, title) in enumerate([(np.array([0.3, 0.6, 1e-9]), "Elbow singularity (q3 = 0)"),
                                (sing_qs[1], "Shoulder singularity (hand on abduction axis)")]):
    q = np.asarray(q)
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    draw_arm(ax, q, "b")
    U, s, Vt = np.linalg.svd(arm.jacobian_pos(q, world=True))
    pe = arm.fk(q, world=True)
    ax.quiver(*pe, *(0.25 * U[:, -1]), color="r", lw=2, label="lost Cartesian direction")
    ax0 = P.shoulder_pos
    ax.plot([ax0[0], ax0[0] + 0.8], [ax0[1]] * 2, [ax0[2]] * 2, "k:", label="q1 axis (anterior)")
    ax.set_xlim(-0.2, 0.8); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0.6, 1.6)
    ax.set_title(f"{title}\n sigma_min = {s[-1]:.2e}"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "singular_configurations.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------- 5. resolved-rate through a near-singularity
# Straight-line hand motion across the shoulder abduction axis, 2 cm above it.
# (pure kinematic demo: joint limits are not enforced here)
pA = P.shoulder_pos + np.array([0.45, 0.25, 0.02])
pB = P.shoulder_pos + np.array([0.45, -0.25, 0.02])
T, dt = 3.0, 1e-3
t = np.arange(0, T + dt, dt)
s = 10 * (t / T) ** 3 - 15 * (t / T) ** 4 + 6 * (t / T) ** 5
sd = (30 * (t / T) ** 2 - 60 * (t / T) ** 3 + 30 * (t / T) ** 4) / T
pd = pA + s[:, None] * (pB - pA)
vd = sd[:, None] * (pB - pA)
q_start = min(arm.ik(pA, within_limits=False), key=lambda q: abs(q[0]))
results = {}
for method in ["pinv", "dls"]:
    q = q_start.copy(); qs, qds, err, wv = [], [], [], []
    for k in range(len(t)):
        J = arm.jacobian_pos(q, world=True)
        e = pd[k] - arm.fk(q, world=True)
        v = vd[k] + 20.0 * e
        if method == "pinv":
            qdot = np.linalg.pinv(J) @ v
        else:  # damped least squares with manipulability-dependent damping
            wm, w0, lam_max = arm.manipulability(q), 0.03, 0.15
            lam2 = lam_max**2 * (1 - (wm / w0) ** 2) if wm < w0 else 0.0
            qdot = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), v)
        qs.append(q.copy()); qds.append(qdot); err.append(np.linalg.norm(e)); wv.append(arm.manipulability(q))
        q = q + dt * qdot
    results[method] = dict(q=np.array(qs), qd=np.array(qds), err=np.array(err), w=np.array(wv))

fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for m, ls in [("pinv", "-"), ("dls", "--")]:
    r = results[m]
    for j, c in enumerate("rgb"):
        axs[0].plot(t, r["qd"][:, j] / DEG, ls, color=c, label=f"{m} qdot{j + 1}")
    axs[1].plot(t, r["err"] * 1000, ls, label=m)
    axs[2].plot(t, r["w"], ls, label=m)
for j in range(3):
    axs[0].axhline(P.qdot_max[j] / DEG, color="gray", lw=0.5); axs[0].axhline(-P.qdot_max[j] / DEG, color="gray", lw=0.5)
axs[0].set_ylabel("joint rate [deg/s]"); axs[0].legend(ncol=2, fontsize=8)
axs[1].set_ylabel("hand error [mm]"); axs[1].legend()
axs[2].set_ylabel("manipulability"); axs[2].set_xlabel("time [s]"); axs[2].legend()
axs[0].set_title("Resolved-rate control 2 cm from the shoulder singularity: pseudo-inverse vs DLS")
fig.tight_layout(); fig.savefig(OUT / "singularity_resolved_rate.png", dpi=150); plt.close(fig)
summary["resolved_rate_near_singularity"] = {
    m: {"peak_joint_rate_deg_s": (np.abs(r["qd"]).max(0) / DEG).round(1).tolist(),
        "max_hand_error_mm": float(r["err"].max() * 1000),
        "exceeds_qdot_max": bool(np.any(np.abs(r["qd"]) > P.qdot_max))}
    for m, r in results.items()}

# ------------------------------------------------------------- 6. task-space path of the rehab task
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
pt = arm.fk_batch(qt)
ax.plot(*pt.T, "k-", lw=2, label="hand path")
for name, q in zip(traj.names, traj.q):
    draw_arm(ax, q, lw=2)
    ax.text(*arm.fk(q, world=True), name, fontsize=8)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title("Rehabilitation task (min-jerk waypoints)")
ax.set_box_aspect((1, 1, 1))
fig.tight_layout(); fig.savefig(OUT / "rehab_task_path.png", dpi=150); plt.close(fig)
summary["rehab_task_min_manipulability"] = float(arm.manipulability(qt).min())

(OUT / "kinematics_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
