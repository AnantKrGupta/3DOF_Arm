"""Denavit-Hartenberg kinematics of the 3-DOF rehab arm.

Kinematic structure (standard DH, "anthropomorphic / elbow" RRR arm):

    i | a_i | alpha_i | d_i | theta_i
    --+-----+---------+-----+---------
    1 |  0  |  +90 deg|  0  |  q1   shoulder abduction / adduction
    2 |  L1 |    0    |  0  |  q2   shoulder flexion / extension
    3 |  L2 |    0    |  0  |  q3   elbow flexion

DH base frame {0} sits at the shoulder centre and is mounted so that
    z0 = +X_world (anterior / forward), x0 = -Z_world (down), y0 = +Y_world (left)
so q = 0 is the arm hanging straight down (anatomical neutral) and
    q1 > 0 abducts the (left) arm sideways, q2 > 0 flexes it forward,
    q3 > 0 flexes the elbow.

Closed form in frame {0}, with A = L1 c2 + L2 c23:
    p0 = [c1 A, s1 A, L1 s2 + L2 s23]
    det(Jv) = -L1 L2 s3 (L1 c2 + L2 c23)
Singularities: s3 = 0 (elbow straight -> workspace boundary) and
A = 0 (hand on the shoulder abduction axis -> q1 has no effect).
"""
from __future__ import annotations

import numpy as np

from .params import ArmParams, load_params

# Columns = x0, y0, z0 expressed in world coordinates.
R_W0 = np.array([[0.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0],
                 [-1.0, 0.0, 0.0]])


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class ThreeDofArm:
    def __init__(self, params: ArmParams | None = None):
        self.p = params or load_params()
        self.L1, self.L2 = self.p.L1, self.p.L2
        self.T_w0 = np.eye(4)
        self.T_w0[:3, :3] = R_W0
        self.T_w0[:3, 3] = self.p.shoulder_pos

    # ------------------------------------------------------------------ DH
    def dh_table(self, q):
        q = np.asarray(q, float)
        return np.array([[0.0, np.pi / 2, 0.0, q[0]],
                         [self.L1, 0.0, 0.0, q[1]],
                         [self.L2, 0.0, 0.0, q[2]]])

    def fk_all(self, q, world: bool = False):
        """Return [T_0^0, T_0^1, T_0^2, T_0^3] (or world-referenced)."""
        T = self.T_w0.copy() if world else np.eye(4)
        out = [T.copy()]
        for a, al, d, th in self.dh_table(q):
            T = T @ dh_transform(a, al, d, th)
            out.append(T.copy())
        return out

    def fk(self, q, world: bool = False) -> np.ndarray:
        return self.fk_all(q, world)[-1][:3, 3]

    def fk_batch(self, Q, world: bool = True) -> np.ndarray:
        """Vectorised closed-form FK, Q shape (N,3) -> (N,3)."""
        Q = np.atleast_2d(Q)
        q1, q2, q3 = Q[:, 0], Q[:, 1], Q[:, 2]
        A = self.L1 * np.cos(q2) + self.L2 * np.cos(q2 + q3)
        p0 = np.stack([np.cos(q1) * A, np.sin(q1) * A,
                       self.L1 * np.sin(q2) + self.L2 * np.sin(q2 + q3)], axis=1)
        return p0 @ R_W0.T + self.p.shoulder_pos if world else p0

    def elbow_batch(self, Q, world: bool = True) -> np.ndarray:
        Q = np.atleast_2d(Q)
        q1, q2 = Q[:, 0], Q[:, 1]
        p0 = self.L1 * np.stack([np.cos(q1) * np.cos(q2), np.sin(q1) * np.cos(q2),
                                 np.sin(q2)], axis=1)
        return p0 @ R_W0.T + self.p.shoulder_pos if world else p0

    def keypoints(self, q, world: bool = True) -> np.ndarray:
        """Shoulder, elbow, hand positions (3,3) - used for plots & safety."""
        Ts = self.fk_all(q, world)
        return np.array([Ts[1][:3, 3], Ts[2][:3, 3], Ts[3][:3, 3]])

    # ------------------------------------------------------------ Jacobian
    def jacobian(self, q, world: bool = False) -> np.ndarray:
        """6x3 geometric Jacobian [Jv; Jw]."""
        Ts = self.fk_all(q, world)
        pe = Ts[-1][:3, 3]
        J = np.zeros((6, 3))
        for i in range(3):
            z, o = Ts[i][:3, 2], Ts[i][:3, 3]
            J[:3, i] = np.cross(z, pe - o)
            J[3:, i] = z
        return J

    def jacobian_pos(self, q, world: bool = False) -> np.ndarray:
        return self.jacobian(q, world)[:3]

    def det_jv(self, q) -> float:
        q = np.asarray(q, float)
        return -self.L1 * self.L2 * np.sin(q[..., 2]) * (
            self.L1 * np.cos(q[..., 1]) + self.L2 * np.cos(q[..., 1] + q[..., 2]))

    def manipulability(self, q):
        """Yoshikawa measure sqrt(det(Jv Jv^T)) = |det Jv| for a square Jv."""
        return np.abs(self.det_jv(q))

    def condition_number(self, q) -> float:
        s = np.linalg.svd(self.jacobian_pos(q), compute_uv=False)
        return np.inf if s[-1] < 1e-12 else s[0] / s[-1]

    # ------------------------------------------------------------------ IK
    def in_limits(self, q, tol: float = 1e-9) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.p.limits[:, 0] - tol) and np.all(q <= self.p.limits[:, 1] + tol))

    def ik(self, p, world: bool = True, within_limits: bool = True):
        """Analytic IK. Returns a list of up to 4 solutions (shoulder x elbow)."""
        p = np.asarray(p, float)
        p0 = R_W0.T @ (p - self.p.shoulder_pos) if world else p
        x, y, z = p0
        sols = []
        for sgn in (+1.0, -1.0):
            A = sgn * np.hypot(x, y)
            q1 = np.arctan2(sgn * y, sgn * x) if np.hypot(x, y) > 1e-12 else 0.0
            D = (A * A + z * z - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
            if abs(D) > 1.0 + 1e-12:
                continue
            D = np.clip(D, -1.0, 1.0)
            for el in (+1.0, -1.0):
                q3 = np.arctan2(el * np.sqrt(1 - D * D), D)
                q2 = np.arctan2(z, A) - np.arctan2(self.L2 * np.sin(q3), self.L1 + self.L2 * np.cos(q3))
                q = wrap(np.array([q1, q2, q3]))
                if within_limits and not self.in_limits(q):
                    continue
                if not any(np.allclose(q, s, atol=1e-9) for s in sols):
                    sols.append(q)
        return sols

    def ik_closest(self, p, q_ref, world: bool = True):
        sols = self.ik(p, world)
        if not sols:
            return None
        return min(sols, key=lambda s: np.linalg.norm(s - q_ref))

    # ------------------------------------------------------------ sampling
    def sample_joint_space(self, n: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        lo, hi = self.p.limits[:, 0], self.p.limits[:, 1]
        return lo + (hi - lo) * rng.random((n, 3))


# ---------------------------------------------------------------- geometry
def point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def arm_obstacle_distance(arm: ThreeDofArm, q, obstacle_center, obstacle_radius: float = 0.0):
    """Minimum surface distance between a spherical obstacle and the arm links
    modelled as capsules (upper arm radius r1, forearm radius r2)."""
    s, e, h = arm.keypoints(q, world=True)
    c = np.asarray(obstacle_center, float)
    d_upper = point_segment_distance(c, s, e) - arm.p.r1
    d_fore = point_segment_distance(c, e, h) - arm.p.r2
    return min(d_upper, d_fore) - obstacle_radius
