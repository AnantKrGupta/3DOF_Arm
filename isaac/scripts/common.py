"""Shared helpers for the Isaac Sim digital twin (tolerant of Isaac Sim 4.5 / 5.x APIs).

Import order matters in Isaac Sim: call `start_app()` BEFORE importing any
omni / isaacsim / pxr-dependent helper below.
"""
from __future__ import annotations

import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
ASSET_USD = REPO / "isaac" / "assets" / "rehab_arm.usda"
ARM_PATH = "/World/rehab_arm"
OUT_DIR = REPO / "isaac" / "outputs"


def start_app(headless: bool = True, width: int = 1280, height: int = 720):
    from isaacsim import SimulationApp
    return SimulationApp({"headless": headless, "width": width, "height": height,
                          "anti_aliasing": 0, "renderer": "RaytracedLighting"})


def core_api():
    """Return (World, Articulation, add_reference_to_stage, ArticulationAction)."""
    try:  # Isaac Sim >= 4.5
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation as Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:  # Isaac Sim <= 4.2
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.types import ArticulationAction
    return World, Articulation, add_reference_to_stage, ArticulationAction


def enable_extension(name: str):
    try:
        from isaacsim.core.utils.extensions import enable_extension as _en
    except ImportError:
        from omni.isaac.core.utils.extensions import enable_extension as _en
    _en(name)


def get_stage():
    import omni.usd
    return omni.usd.get_context().get_stage()


# ----------------------------------------------------------------- semantics
def add_semantic(prim, label: str):
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[label], instance_name="class")
        return
    except Exception:
        pass
    try:
        try:
            from isaacsim.core.utils.semantics import add_update_semantics
        except ImportError:
            from omni.isaac.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, label)
        return
    except Exception:
        pass
    from pxr import Semantics
    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set(label)


# --------------------------------------------------------------------- scene
def _shape(stage, path, kind, pos, color, size=None, radius=None, height=None, axis="Z", collide=False):
    from pxr import Gf, UsdGeom, UsdPhysics
    if kind == "cube":
        g = UsdGeom.Cube.Define(stage, path); g.CreateSizeAttr(1.0)
    elif kind == "sphere":
        g = UsdGeom.Sphere.Define(stage, path); g.CreateRadiusAttr(radius)
    elif kind == "capsule":
        g = UsdGeom.Capsule.Define(stage, path); g.CreateRadiusAttr(radius); g.CreateHeightAttr(height)
        g.CreateAxisAttr(axis)
    else:
        g = UsdGeom.Cylinder.Define(stage, path); g.CreateRadiusAttr(radius); g.CreateHeightAttr(height)
        g.CreateAxisAttr(axis)
    xf = UsdGeom.Xformable(g)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if size is not None:
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
    g.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    return g


def build_scene(stage, params, add_reference_to_stage):
    """Clinic-like scene: floor, patient torso/head, arm stand, table, cup,
    the imported USD arm and a 'human hand' obstacle."""
    from pxr import Gf, UsdGeom, UsdLux

    sx, sy, sz = params.shoulder_pos
    UsdGeom.Xform.Define(stage, "/World")
    _shape(stage, "/World/Ground", "cube", (0, 0, -0.01), (0.55, 0.57, 0.6), size=(8, 8, 0.02), collide=True)
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight"); dome.CreateIntensityAttr(900.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/KeyLight"); sun.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, 20, 30))

    # patient (left arm = the robot arm), visual only
    _shape(stage, "/World/Patient/torso", "capsule", (sx - 0.02, sy - 0.20, sz - 0.30), (0.75, 0.55, 0.45),
           radius=0.14, height=0.40)
    _shape(stage, "/World/Patient/head", "sphere", (sx - 0.02, sy - 0.20, sz + 0.25), (0.85, 0.7, 0.6), radius=0.10)
    _shape(stage, "/World/Patient/chair", "cube", (sx - 0.05, sy - 0.2, 0.45), (0.25, 0.25, 0.3), size=(0.5, 0.5, 0.06))
    # exoskeleton/arm support stand behind the shoulder
    _shape(stage, "/World/Stand/post", "cylinder", (sx - 0.15, sy + 0.05, sz / 2), (0.3, 0.3, 0.32),
           radius=0.03, height=sz)
    _shape(stage, "/World/Stand/boom", "cube", (sx - 0.105, sy + 0.05, sz), (0.3, 0.3, 0.32), size=(0.09, 0.04, 0.04))
    # rehab table + cup target
    _shape(stage, "/World/Table/top", "cube", (sx + 0.65, sy, 0.72), (0.6, 0.45, 0.3), size=(0.5, 0.9, 0.04))
    _shape(stage, "/World/Cup", "cylinder", (sx + 0.55, sy + 0.12, 0.79), (0.9, 0.2, 0.2), radius=0.035, height=0.10)

    add_reference_to_stage(usd_path=str(ASSET_USD), prim_path=ARM_PATH)
    arm_prim = stage.GetPrimAtPath(ARM_PATH)
    UsdGeom.Xformable(arm_prim).AddTranslateOp().Set(Gf.Vec3d(sx, sy, sz))

    r = params.safety.get("obstacle_radius_m", 0.05)
    hand = _shape(stage, "/World/HumanHand", "sphere", (sx + 1.0, sy - 0.5, sz - 0.2), (0.1, 0.8, 0.2), radius=r)

    for path, label in [(f"{ARM_PATH}/upper_arm", "upper_arm"), (f"{ARM_PATH}/forearm", "forearm"),
                        (f"{ARM_PATH}/forearm/hand", "robot_hand"), ("/World/HumanHand", "human_hand"),
                        ("/World/Patient", "patient"), ("/World/Table", "table"), ("/World/Cup", "cup")]:
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            add_semantic(p, label)
    return hand


def set_translate(prim, pos):
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(prim)
    ops = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    (ops[0] if ops else xf.AddTranslateOp()).Set(Gf.Vec3d(*map(float, pos)))


def set_color(prim, rgb):
    from pxr import Gf, UsdGeom
    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])


def make_camera(stage, path, eye, target, focal=18.0):
    from pxr import UsdGeom
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr().Set((0.01, 100.0))
    look_at(stage, path, eye, target)
    return cam


def look_at(stage, path, eye, target):
    from pxr import Gf, UsdGeom
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())


def world_position_reader(prim_path: str):
    """Return a callable giving the world position of a prim (fabric-aware)."""
    try:
        from isaacsim.core.prims import SingleXFormPrim
        xp = SingleXFormPrim(prim_path)
        return lambda: xp.get_world_pose()[0]
    except Exception:
        pass
    try:
        from omni.isaac.core.prims import XFormPrim
        xp = XFormPrim(prim_path)
        return lambda: xp.get_world_pose()[0]
    except Exception:
        pass
    from pxr import Usd, UsdGeom
    prim = get_stage().GetPrimAtPath(prim_path)
    return lambda: list(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        .ExtractTranslation())


def deg(x):
    return x * 180.0 / math.pi
