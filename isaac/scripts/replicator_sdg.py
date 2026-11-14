"""Replicator synthetic-data generation for the rehab-arm digital twin.

Each frame randomises: arm joint configuration (within limits), human-hand
obstacle position, camera viewpoint, light intensity/colour/direction.
Annotations (Replicator BasicWriter): RGB, depth (distance_to_camera),
semantic segmentation, tight 2D boxes. Additionally writes labels.csv with
joint angles, 3D keypoints, 2D projected keypoints and arm-hand separation
(useful for pose-estimation / proximity-perception training).

Run (headless):  python isaac/scripts/replicator_sdg.py --frames 300
Output -> isaac/outputs/sdg/
"""
import argparse
import csv
import json

import numpy as np

from common import OUT_DIR, ARM_PATH, start_app

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--seed", type=int, default=0)
args, _ = ap.parse_known_args()

app = start_app(headless=True)

from common import (core_api, build_scene, get_stage, set_translate, make_camera, look_at,  # noqa: E402
                    enable_extension)
from rehab_arm import ThreeDofArm, load_params, arm_obstacle_distance  # noqa: E402

enable_extension("omni.replicator.core")
import omni.replicator.core as rep  # noqa: E402
from pxr import Gf, UsdLux  # noqa: E402

World, Articulation, add_reference_to_stage, ArticulationAction = core_api()
OUT = OUT_DIR / "sdg"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

params = load_params()
kin = ThreeDofArm(params)
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
stage = get_stage()
hand_prim = build_scene(stage, params, add_reference_to_stage).GetPrim()
arm = Articulation(prim_path=ARM_PATH, name="rehab_arm")
world.scene.add(arm)

S = np.asarray(params.shoulder_pos)
CAM = "/World/SDGCamera"
FOCAL = 18.0
make_camera(stage, CAM, eye=S + [1.6, 1.0, 0.3], target=S + [0.2, 0, -0.2], focal=FOCAL)
try:
    rep.orchestrator.set_capture_on_play(False)
except Exception:
    pass
rp = rep.create.render_product(CAM, (args.width, args.height))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=str(OUT), rgb=True, distance_to_camera=True, semantic_segmentation=True,
                  colorize_semantic_segmentation=True, bounding_box_2d_tight=True)
writer.attach([rp])

world.reset()
names = list(arm.dof_names)
idx = np.array([names.index(n) for n in params.joint_names])
dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
key = UsdLux.DistantLight(stage.GetPrimAtPath("/World/KeyLight"))


def full(v):
    out = np.zeros(arm.num_dof); out[idx] = v; return out


def project(points, eye, target):
    """Pinhole projection matching the USD camera (default 20.955 mm horizontal aperture)."""
    view = np.array(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))).T
    ph = np.c_[points, np.ones(len(points))] @ view.T
    xc, yc, zc = ph[:, 0], ph[:, 1], -ph[:, 2]
    fpx = FOCAL / 20.955 * args.width
    u = args.width / 2 + fpx * xc / zc
    v = args.height / 2 - fpx * yc / zc
    return np.c_[u, v]


rows = []
lo, hi = params.limits[:, 0], params.limits[:, 1]
for i in range(args.frames):
    q = lo + (hi - lo) * rng.random(3)
    arm.set_joint_positions(full(q)); arm.set_joint_velocities(np.zeros(arm.num_dof))
    arm.apply_action(ArticulationAction(joint_positions=full(q)))
    # obstacle: 50 % near the arm (proximity cases), 50 % anywhere in front
    kp = kin.keypoints(q)
    if rng.random() < 0.5:
        c = kp[rng.integers(1, 3)] + rng.normal(0, 0.12, 3)
    else:
        c = S + rng.uniform([0.1, -0.6, -0.6], [1.0, 0.7, 0.5])
    set_translate(hand_prim, c)
    # camera on a randomised hemisphere around the arm
    az, el, rad = rng.uniform(-0.3, 2.2), rng.uniform(-0.15, 0.6), rng.uniform(1.4, 2.4)
    target = S + [0.2, 0.0, -0.2] + rng.normal(0, 0.05, 3)
    eye = target + rad * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    look_at(stage, CAM, eye, target)
    dome.GetIntensityAttr().Set(float(rng.uniform(300, 1500)))
    key.GetIntensityAttr().Set(float(rng.uniform(800, 4000)))
    key.GetColorAttr().Set(Gf.Vec3f(*rng.uniform(0.8, 1.0, 3)))
    for _ in range(2):
        world.step(render=True)
    try:
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
    except TypeError:
        rep.orchestrator.step(rt_subframes=4)
    qm = np.asarray(arm.get_joint_positions())[idx]
    kpm = kin.keypoints(qm)
    uv = project(kpm, eye, target)
    d = arm_obstacle_distance(kin, qm, c, params.safety["obstacle_radius_m"])
    rows.append([i, *qm, *kpm.ravel(), *uv.ravel(), *c, d, *eye])
    if i % 25 == 0:
        print(f"[sdg] frame {i}/{args.frames}  d={d:.3f}")

try:
    rep.orchestrator.wait_until_complete()
except Exception:
    pass
hdr = (["frame", "q1", "q2", "q3"] + [f"{k}_{a}" for k in ("shoulder", "elbow", "hand") for a in "xyz"]
       + [f"{k}_{a}" for k in ("shoulder", "elbow", "hand") for a in "uv"]
       + ["obst_x", "obst_y", "obst_z", "separation_m", "cam_x", "cam_y", "cam_z"])
with open(OUT / "labels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
(OUT / "dataset_info.json").write_text(json.dumps({
    "frames": args.frames, "resolution": [args.width, args.height],
    "annotations": ["rgb", "distance_to_camera", "semantic_segmentation", "bounding_box_2d_tight", "labels.csv"],
    "classes": ["upper_arm", "forearm", "robot_hand", "human_hand", "patient", "table", "cup"],
    "randomised": ["joint angles", "obstacle position", "camera pose", "light intensity/colour"],
    "proximity_fraction_d<0.10m": float(np.mean(np.array([r[-4] for r in rows]) < 0.10))}, indent=2))
print("[sdg] done ->", OUT)
writer.detach()
app.close()
