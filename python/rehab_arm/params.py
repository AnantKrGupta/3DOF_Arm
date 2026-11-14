"""Load the single-source-of-truth arm parameters (config/arm_params.json)."""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "arm_params.json"

# Order of the dynamic parameter vector used by the generated M/C/G code
# (Python AND MATLAB/Simulink use exactly this order).
DYN_PARAM_NAMES = ["L1", "L2", "c1", "c2", "m1", "m2", "mp", "r1", "r2", "g"]


@dataclass
class ArmParams:
    L1: float
    L2: float
    r1: float
    r2: float
    m1: float
    m2: float
    mp: float
    c1: float
    c2: float
    g: float
    joint_names: list
    limits: np.ndarray            # (3, 2) rad
    qdot_max: np.ndarray          # (3,) rad/s
    tau_max: np.ndarray           # (3,) Nm
    viscous: np.ndarray           # (3,) Nm s/rad
    shoulder_pos: np.ndarray      # (3,) world, m
    safety: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def dyn_vector(self, payload: float | None = None) -> np.ndarray:
        mp = self.mp if payload is None else payload
        return np.array([self.L1, self.L2, self.c1, self.c2, self.m1,
                         self.m2, mp, self.r1, self.r2, self.g], dtype=float)

    @property
    def reach(self) -> float:
        return self.L1 + self.L2


def load_params(path: str | pathlib.Path | None = None) -> ArmParams:
    path = pathlib.Path(path) if path else CONFIG_PATH
    d = json.loads(path.read_text())
    deg = math.pi / 180.0
    return ArmParams(
        L1=d["links"]["L1_upper_arm_m"],
        L2=d["links"]["L2_forearm_hand_m"],
        r1=d["links"]["r1_upper_arm_radius_m"],
        r2=d["links"]["r2_forearm_radius_m"],
        m1=d["mass"]["m1_upper_arm_kg"],
        m2=d["mass"]["m2_forearm_kg"],
        mp=d["mass"]["mp_payload_kg"],
        c1=d["com_ratio"]["c1_upper_arm"],
        c2=d["com_ratio"]["c2_forearm"],
        g=d["gravity_m_s2"],
        joint_names=list(d["joint_names"]),
        limits=np.asarray(d["joint_limits_deg"], float) * deg,
        qdot_max=np.asarray(d["qdot_max_deg_s"], float) * deg,
        tau_max=np.asarray(d["tau_max_Nm"], float),
        viscous=np.asarray(d["viscous_Nms_rad"], float),
        shoulder_pos=np.asarray(d["world"]["shoulder_position_m"], float),
        safety=dict(d.get("safety", {})),
        raw=d,
    )
