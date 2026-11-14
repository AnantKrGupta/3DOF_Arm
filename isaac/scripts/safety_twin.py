"""Isaac Sim digital twin: USD import, twin validation and proximity-based safety.

  1. Twin validation : teleport to test poses, compare Isaac end-effector
                       position with the Python DH FK; hold poses with and
                       without Lagrangian gravity compensation G(q).
  2. Safety demo     : the arm loops the rehab task while a simulated human
                       hand approaches. Speed & separation monitoring
                       (ISO/TS 15066 style) time-scales the trajectory:
                         d > d_slow            -> alpha = 1      (RUN)
                         d_stop < d <= d_slow  -> alpha linear   (SLOW)
                         d <= d_stop or overlap-> alpha = 0      (PROTECTIVE STOP)
                         resume only when d > d_resume (hysteresis)
                       Channel A: analytic capsule distance from measured q.
                       Channel B: PhysX overlap-sphere query (redundant).

Run (headless):  python isaac/scripts/safety_twin.py [--duration 26] [--no-video]
Outputs -> isaac/outputs/safety/{twin_log.csv, validation.json, summary.json, safety_demo.mp4}
"""
import argparse
import csv
import json
import time

import numpy as np

from common import OUT_DIR, ARM_PATH, start_app

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=26.0)
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--gui", action="store_true", help="run with GUI (local workstation only)")
args, _ = ap.parse_known_args()

app = start_app(headless=not args.gui)

# ---- imports that need the running app
from common import (core_api, build_scene, get_stage, set_translate, set_color, make_camera,  # noqa: E402
                    world_position_reader, deg)
from rehab_arm import ThreeDofArm, WaypointTrajectory, load_params, arm_obstacle_distance  # noqa: E402
from rehab_arm.dynamics import ArmDynamics  # noqa: E402

World, Articulation, add_reference_to_stage, ArticulationAction = core_api()
OUT = OUT_DIR / "safety"
OUT.mkdir(parents=True, exist_ok=True)

params = load_params()
kin = ThreeDofArm(params)
dyn = ArmDynamics(params)
traj = WaypointTrajectory()
S = params.safety
d_slow, d_stop, d_resume = S["d_slow_m"], S["d_stop_m"], S["d_resume_m"]
r_obs = S["obstacle_radius_m"]
alpha_rate = S.get("alpha_rate_limit_per_s", 4.0)

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
stage = get_stage()
hand_prim = build_scene(stage, params, add_reference_to_stage).GetPrim()
arm = Articulation(prim_path=ARM_PATH, name="rehab_arm")
world.scene.add(arm)

# camera + RGB annotator for the video
frames = []
rgb = None
if not args.no_video:
    try:
        import omni.replicator.core as rep
        sx, sy, sz = params.shoulder_pos
        make_camera(stage, "/World/VideoCam", eye=(sx + 1.7, sy + 1.3, sz + 0.45), target=(sx + 0.25, sy, sz - 0.2))
        rp = rep.create.render_product("/World/VideoCam", (960, 540))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])
    except Exception as e:  # pragma: no cover
        print("[twin] video disabled:", e)

world.reset()
names = list(arm.dof_names)
idx = np.array([names.index(n) for n in params.joint_names])
print("[twin] DOFs:", names, "-> DH order index", idx)
ee_reader = world_position_reader(f"{ARM_PATH}/forearm/ee_frame")

try:
    from omni.physx import get_physx_scene_query_interface
    import carb
    sq = get_physx_scene_query_interface()
except Exception:
    sq = None


def overlap_hits(center, radius):
    if sq is None:
        return False
    hits = []

    def cb(hit):
        if str(hit.rigid_body).startswith(ARM_PATH):
            hits.append(hit.rigid_body)
        return True
    sq.overlap_sphere(float(radius), carb.Float3(*map(float, center)), cb, False)
    return len(hits) > 0


def full(v):
    out = np.zeros(arm.num_dof)
    out[idx] = v
    return out


def get_q():
    return np.asarray(arm.get_joint_positions())[idx]


def command(q_t, qd_t=None, grav=True):
    q_meas = get_q()
    arm.apply_action(ArticulationAction(joint_positions=full(q_t),
                                        joint_velocities=None if qd_t is None else full(qd_t)))
    arm.set_joint_efforts(full(dyn.G(q_meas)) if grav else np.zeros(arm.num_dof))


# ======================================================= 1. twin validation
val = {"poses": []}
test_poses = np.deg2rad([[0, 0, 0], [30, 45, 60], [10, 90, 0], [60, 30, 100], [20, 55, 125], [120, 10, 30]])
for qv in test_poses:
    entry = {"q_deg": np.rad2deg(qv).round(1).tolist()}
    for grav in (False, True):
        arm.set_joint_positions(full(qv)); arm.set_joint_velocities(np.zeros(arm.num_dof))
        for _ in range(90):             # hold 1.5 s
            command(qv, np.zeros(3), grav)
            world.step(render=False)
        qm = get_q()
        p_isaac = np.asarray(ee_reader(), float)
        p_fk = kin.fk(qm, world=True)
        entry["gravity_comp" if grav else "no_gravity_comp"] = {
            "steady_err_deg": np.rad2deg(qv - qm).round(4).tolist(),
            "ee_isaac_vs_fk_mm": float(np.linalg.norm(p_isaac - p_fk) * 1000)}
    val["poses"].append(entry)
    print("[twin] validation", entry)
val["max_ee_mismatch_mm"] = max(max(e["gravity_comp"]["ee_isaac_vs_fk_mm"], e["no_gravity_comp"]["ee_isaac_vs_fk_mm"])
                                for e in val["poses"])
(OUT / "validation.json").write_text(json.dumps(val, indent=2))

# ======================================================= 2. safety demo
sx, sy, sz = params.shoulder_pos
FAR = np.array([sx + 1.0, sy - 0.55, sz - 0.15])
# behaviour script of the human hand: (t_end, mode, speed m/s, goal distance m)
SCRIPT = [(3.0, "far", 0.0, 0.0),
          (8.0, "approach", 0.15, 0.20),   # slow approach, stays in SLOW zone
          (10.0, "retreat", 0.30, 0.0),
          (15.0, "approach", 0.20, 0.04),  # reaches into the arm -> protective stop
          (18.0, "retreat", 0.25, 0.0),
          (22.0, "approach", 0.60, 0.06),  # fast intrusion
          (1e9, "retreat", 0.30, 0.0)]

arm.set_joint_positions(full(traj.q[0])); arm.set_joint_velocities(np.zeros(arm.num_dof))
h = FAR.copy()
s_task, alpha, state = 0.0, 1.0, "RUN"
dt = 1.0 / 60.0
n_steps = int(args.duration / dt)
log = []
t0 = time.time()
for k in range(n_steps):
    t = k * dt
    q = get_q()
    # --- human hand behaviour
    mode, speed, goal = next((m, v, g) for (te, m, v, g) in SCRIPT if t < te)
    keypts = kin.keypoints(q)
    if mode == "far":
        h = FAR.copy()
    elif mode == "approach":
        # head towards the closest point on the forearm/upper-arm segments
        seg = [(keypts[0], keypts[1]), (keypts[1], keypts[2])]
        cands = []
        for a, b in seg:
            ab = b - a
            tt = np.clip(np.dot(h - a, ab) / np.dot(ab, ab), 0, 1)
            cands.append(a + tt * ab)
        target = min(cands, key=lambda c: np.linalg.norm(c - h))
        if arm_obstacle_distance(kin, q, h, r_obs) > goal:
            v = target - h
            h = h + speed * dt * v / max(np.linalg.norm(v), 1e-9)
    else:
        v = FAR - h
        if np.linalg.norm(v) > 1e-3:
            h = h + min(speed * dt, np.linalg.norm(v)) * v / np.linalg.norm(v)
    set_translate(hand_prim, h)

    # --- safety monitor (two channels)
    d = arm_obstacle_distance(kin, q, h, r_obs)
    hit = overlap_hits(h, r_obs + d_stop)
    if state == "STOP":
        if d > d_resume and not hit:
            state = "SLOW" if d <= d_slow else "RUN"
    elif d <= d_stop or hit:
        state = "STOP"
    else:
        state = "SLOW" if d <= d_slow else "RUN"
    a_target = {"RUN": 1.0, "STOP": 0.0}.get(state, float(np.clip((d - d_stop) / (d_slow - d_stop), 0, 1)))
    rate = alpha_rate * (3.0 if a_target < alpha else 1.0)   # decelerate faster than accelerate
    alpha += float(np.clip(a_target - alpha, -rate * dt, rate * dt))
    set_color(hand_prim, {"RUN": (0.1, 0.8, 0.2), "SLOW": (0.95, 0.75, 0.05), "STOP": (0.9, 0.1, 0.1)}[state])

    # --- time-scaled trajectory + gravity compensation
    s_task = (s_task + alpha * dt) % traj.duration
    q_ref, qd_ref, _ = traj(s_task)
    command(q_ref, alpha * qd_ref, grav=True)
    world.step(render=rgb is not None)

    p_isaac = np.asarray(ee_reader(), float)
    p_fk = kin.fk(get_q(), world=True)
    log.append([t, s_task, alpha, state, d, int(hit), *q, *q_ref, *h, *p_isaac,
                float(np.linalg.norm(p_isaac - p_fk))])
    if rgb is not None and k % 2 == 0 and k > 4:
        img = rgb.get_data()
        if img is not None and getattr(img, "size", 0) > 0:
            frames.append((np.asarray(img)[..., :3].copy(), state, d, alpha, t))
    if k % 120 == 0:
        print(f"[twin] t={t:5.1f}s state={state:4s} d={d:.3f} alpha={alpha:.2f}")

hdr = ["t", "s_task", "alpha", "state", "d_min", "overlap_hit", "q1", "q2", "q3", "q1_ref", "q2_ref", "q3_ref",
       "hx", "hy", "hz", "ee_x", "ee_y", "ee_z", "ee_twin_err_m"]
with open(OUT / "twin_log.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(log)

states = np.array([r[3] for r in log])
dmin = np.array([r[4] for r in log])
stops = int(np.sum((states[1:] == "STOP") & (states[:-1] != "STOP")))
summary = {
    "duration_s": args.duration, "wall_time_s": round(time.time() - t0, 1),
    "time_in_RUN_s": float(np.sum(states == "RUN") * dt), "time_in_SLOW_s": float(np.sum(states == "SLOW") * dt),
    "time_in_STOP_s": float(np.sum(states == "STOP") * dt), "protective_stops": stops,
    "min_separation_m": float(dmin.min()), "overlap_channel_available": sq is not None,
    "overlap_hits": int(sum(r[5] for r in log)),
    "max_ee_twin_error_mm": float(max(r[-1] for r in log) * 1000),
    "thresholds_m": {"d_slow": d_slow, "d_stop": d_stop, "d_resume": d_resume},
    "validation_max_ee_mismatch_mm": val["max_ee_mismatch_mm"],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("[twin] summary", json.dumps(summary, indent=2))

if frames:
    try:
        import imageio.v2 as imageio
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Image = None
        writer = imageio.get_writer(str(OUT / "safety_demo.mp4"), fps=30, quality=8)
        for img, st, d, a, t in frames:
            if Image is not None:
                im = Image.fromarray(img); dr = ImageDraw.Draw(im)
                col = {"RUN": (40, 200, 60), "SLOW": (240, 190, 20), "STOP": (230, 30, 30)}[st]
                dr.rectangle([10, 10, 380, 92], fill=(0, 0, 0))
                dr.text((20, 18), f"t = {t:5.1f} s   state: {st}", fill=col)
                dr.text((20, 42), f"min separation d = {d * 100:5.1f} cm", fill=(255, 255, 255))
                dr.text((20, 66), f"speed scale alpha = {a:4.2f}", fill=(255, 255, 255))
                img = np.asarray(im)
            writer.append_data(img)
        writer.close()
        print("[twin] video ->", OUT / "safety_demo.mp4")
    except Exception as e:
        print("[twin] could not write mp4 (", e, ") - saving PNG frames instead")
        import imageio.v2 as imageio
        (OUT / "frames").mkdir(exist_ok=True)
        for i, (img, *_r) in enumerate(frames[::3]):
            imageio.imwrite(OUT / "frames" / f"{i:04d}.png", img)

app.close()
