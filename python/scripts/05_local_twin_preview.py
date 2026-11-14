"""Step 5 (local verification) - USD articulation FK check & safety twin preview.

Verifies the exported USD articulation (`isaac/assets/rehab_arm.usda`) directly
with `pxr.Usd` against the Python DH kinematics, then simulates the exact 26 s
ISO/TS 15066 speed-and-separation monitoring scenario using the full Lagrangian
dynamics + USD drive parameters (`DRIVE_KP = 400`, `DRIVE_KD = 30`).

Outputs -> isaac/outputs/safety/{validation.json, twin_log.csv, summary.json,
                                 safety_timeline.png, safety_3d_keyframes.png}
"""
from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rehab_arm import ROOT, ThreeDofArm, WaypointTrajectory, arm_obstacle_distance, load_params  # noqa: E402
from rehab_arm.assets import DRIVE_KD, DRIVE_KP, UsdArticulationFK  # noqa: E402
from rehab_arm.dynamics import ArmDynamics  # noqa: E402

USD_PATH = ROOT / "isaac" / "assets" / "rehab_arm.usda"
OUT = ROOT / "isaac" / "outputs" / "safety"
OUT.mkdir(parents=True, exist_ok=True)


def _sphere_mesh(center, radius, n=18):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def main():
    params = load_params()
    kin = ThreeDofArm(params)
    dyn = ArmDynamics(params)
    traj = WaypointTrajectory()
    usd_fk = UsdArticulationFK(USD_PATH, params)

    # ------------------------------------------------- 1. USD vs DH validation
    test_poses = np.deg2rad([[0, 0, 0], [30, 45, 60], [10, 90, 0],
                            [60, 30, 100], [20, 55, 125], [120, 10, 30]])
    val = {"source": "local_usd_preview", "poses": []}
    for qv in test_poses:
        p_usd = usd_fk.ee_world(qv)
        p_fk = kin.fk(qv, world=True)
        err_mm = float(np.linalg.norm(p_usd - p_fk) * 1000.0)
        assert err_mm < 0.01, f"USD FK mismatch {err_mm:.6f} mm at q={np.rad2deg(qv)}"
        val["poses"].append({
            "q_deg": np.rad2deg(qv).round(1).tolist(),
            "ee_usd_m": p_usd.round(6).tolist(),
            "ee_dh_m": p_fk.round(6).tolist(),
            "ee_usd_vs_fk_mm": err_mm,
        })
    val["max_ee_mismatch_mm"] = max(p["ee_usd_vs_fk_mm"] for p in val["poses"])
    (OUT / "validation.json").write_text(json.dumps(val, indent=2))
    print(f"[local_twin] USD FK validation passed: max mismatch = {val['max_ee_mismatch_mm']:.6e} mm")

    # ------------------------------------------------- 2. 26s safety scenario
    S = params.safety
    d_slow, d_stop, d_resume = S["d_slow_m"], S["d_stop_m"], S["d_resume_m"]
    r_obs = S["obstacle_radius_m"]
    alpha_rate = S.get("alpha_rate_limit_per_s", 4.0)

    sx, sy, sz = params.shoulder_pos
    FAR = np.array([sx + 1.0, sy - 0.55, sz - 0.15])
    SCRIPT = [
        (3.0, "far", 0.0, 0.0),
        (8.0, "approach", 0.15, 0.20),
        (10.0, "retreat", 0.30, 0.0),
        (15.0, "approach", 0.20, 0.04),
        (18.0, "retreat", 0.25, 0.0),
        (22.0, "approach", 0.60, 0.06),
        (1e9, "retreat", 0.30, 0.0),
    ]

    duration = 26.0
    dt = 1.0 / 60.0
    substeps = 16  # 960 Hz RK4 keeps explicit integration stable for DRIVE_KD = 30
    phys_dt = dt / substeps
    n_steps = int(duration / dt)

    q = traj.q[0].copy()
    qd = np.zeros(3)
    h = FAR.copy()
    s_task, alpha, state = 0.0, 1.0, "RUN"
    log = []
    t0 = time.time()

    for k in range(n_steps):
        t = k * dt
        mode, speed, goal = next((m, v, g) for (te, m, v, g) in SCRIPT if t < te)
        keypts = kin.keypoints(q)
        if mode == "far":
            h = FAR.copy()
        elif mode == "approach":
            seg = [(keypts[0], keypts[1]), (keypts[1], keypts[2])]
            cands = []
            for a, b in seg:
                ab = b - a
                tt = np.clip(np.dot(h - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
                cands.append(a + tt * ab)
            target = min(cands, key=lambda c: np.linalg.norm(c - h))
            if arm_obstacle_distance(kin, q, h, r_obs) > goal:
                v = target - h
                h = h + speed * dt * v / max(np.linalg.norm(v), 1e-9)
        else:
            v = FAR - h
            if np.linalg.norm(v) > 1e-3:
                h = h + min(speed * dt, np.linalg.norm(v)) * v / np.linalg.norm(v)

        d = arm_obstacle_distance(kin, q, h, r_obs)
        hit = d <= d_stop
        if state == "STOP":
            if d > d_resume and not hit:
                state = "SLOW" if d <= d_slow else "RUN"
        elif d <= d_stop or hit:
            state = "STOP"
        else:
            state = "SLOW" if d <= d_slow else "RUN"

        a_target = {"RUN": 1.0, "STOP": 0.0}.get(
            state, float(np.clip((d - d_stop) / (d_slow - d_stop), 0.0, 1.0))
        )
        rate = alpha_rate * (3.0 if a_target < alpha else 1.0)
        alpha += float(np.clip(a_target - alpha, -rate * dt, rate * dt))

        s_task = (s_task + alpha * dt) % traj.duration
        q_ref, qd_ref, _ = traj(s_task)

        # USD drive + Lagrangian gravity compensation integrated at 120 Hz
        for _ in range(substeps):
            tau_drive = np.clip(
                DRIVE_KP * (q_ref - q) + DRIVE_KD * (alpha * qd_ref - qd),
                -params.tau_max,
                params.tau_max,
            )
            tau = tau_drive + dyn.G(q)

            def f(x):
                return np.concatenate([x[3:], dyn.forward(x[:3], x[3:], tau)])

            x = np.concatenate([q, qd])
            k1 = f(x)
            k2 = f(x + 0.5 * phys_dt * k1)
            k3 = f(x + 0.5 * phys_dt * k2)
            k4 = f(x + phys_dt * k3)
            x = x + (phys_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            q = np.clip(x[:3], params.limits[:, 0], params.limits[:, 1])
            qd = x[3:]

        p_usd = usd_fk.ee_world(q)
        p_fk = kin.fk(q, world=True)
        log.append([
            t, s_task, alpha, state, d, int(hit),
            *q, *q_ref, *h, *p_usd, float(np.linalg.norm(p_usd - p_fk))
        ])

    hdr = [
        "t", "s_task", "alpha", "state", "d_min", "overlap_hit",
        "q1", "q2", "q3", "q1_ref", "q2_ref", "q3_ref",
        "hx", "hy", "hz", "ee_x", "ee_y", "ee_z", "ee_twin_err_m"
    ]
    with open(OUT / "twin_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(log)

    states = np.array([r[3] for r in log])
    dmin = np.array([r[4] for r in log])
    stops = int(np.sum((states[1:] == "STOP") & (states[:-1] != "STOP")))
    summary = {
        "source": "local_usd_dynamics_preview",
        "duration_s": duration,
        "wall_time_s": round(time.time() - t0, 2),
        "time_in_RUN_s": float(np.sum(states == "RUN") * dt),
        "time_in_SLOW_s": float(np.sum(states == "SLOW") * dt),
        "time_in_STOP_s": float(np.sum(states == "STOP") * dt),
        "protective_stops": stops,
        "min_separation_m": float(dmin.min()),
        "overlap_channel_available": False,
        "overlap_hits": int(sum(r[5] for r in log)),
        "max_ee_twin_error_mm": float(max(r[-1] for r in log) * 1000.0),
        "thresholds_m": {"d_slow": d_slow, "d_stop": d_stop, "d_resume": d_resume},
        "validation_max_ee_mismatch_mm": val["max_ee_mismatch_mm"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[local_twin] summary:", json.dumps(summary, indent=2))

    # ------------------------------------------------- 3. timeline + 3D keyframes
    subprocess.run([sys.executable, str(ROOT / "isaac" / "scripts" / "plot_twin_results.py"), str(OUT)], check=True)

    # Pick representative frames for RUN, SLOW, STOP
    picks = {}
    for target_state, min_t in [("RUN", 1.5), ("SLOW", 5.5), ("STOP", 12.0)]:
        candidates = [idx for idx, r in enumerate(log) if r[3] == target_state and r[0] >= min_t]
        if not candidates:
            candidates = [idx for idx, r in enumerate(log) if r[3] == target_state]
        picks[target_state] = candidates[len(candidates) // 2]

    fig = plt.figure(figsize=(15, 5.2))
    state_colors = {"RUN": "#1f9d3a", "SLOW": "#e69f00", "STOP": "#d62728"}
    for col_idx, st_name in enumerate(["RUN", "SLOW", "STOP"], 1):
        row = log[picks[st_name]]
        t_val, alpha_val, d_val = row[0], row[2], row[4]
        q_val = np.array(row[6:9])
        h_val = np.array(row[12:15])
        kpts = kin.keypoints(q_val)

        ax = fig.add_subplot(1, 3, col_idx, projection="3d")
        ax.plot(kpts[:, 0], kpts[:, 1], kpts[:, 2], "-o", color="#1f77b4", lw=4.5, ms=6, label="rehab arm")
        ax.scatter(*kpts[-1], color="#f2b880", s=70, edgecolor="k", zorder=5, label="robot hand")

        xs, ys, zs = _sphere_mesh(h_val, r_obs)
        ax.plot_surface(xs, ys, zs, color=state_colors[st_name], alpha=0.85, linewidth=0)

        # Safety stop threshold sphere around human hand
        xs2, ys2, zs2 = _sphere_mesh(h_val, r_obs + d_stop, n=14)
        ax.plot_wireframe(xs2, ys2, zs2, color="#d62728", alpha=0.22, lw=0.6)

        # Reference task path
        tt = np.linspace(0, traj.duration, 200)
        qt, _, _ = traj(tt)
        pt = kin.fk_batch(qt)
        ax.plot(pt[:, 0], pt[:, 1], pt[:, 2], "k:", lw=1.0, alpha=0.5, label="task path")

        ax.set_xlim(-0.2, 1.1)
        ax.set_ylim(-0.65, 0.65)
        ax.set_zlim(0.55, 1.75)
        ax.set_box_aspect((1.3, 1.3, 1.2))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(
            f"{st_name} (t = {t_val:.1f} s)\nd = {d_val * 100:.1f} cm, alpha = {alpha_val:.2f}",
            color=state_colors[st_name], fontweight="bold"
        )
        if col_idx == 1:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Digital Twin Proximity Safety Keyframes (USD Kinematics + ISO/TS 15066 SSM)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "safety_3d_keyframes.png", dpi=150)
    plt.close(fig)
    print("saved", OUT / "safety_3d_keyframes.png")


if __name__ == "__main__":
    main()
