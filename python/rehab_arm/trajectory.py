"""Rehabilitation task trajectories (minimum-jerk joint-space waypoints).

The same task definition is mirrored in matlab/rehab_task.m and used by the
Isaac Sim digital twin, so all three studies track the identical motion.
"""
from __future__ import annotations

import numpy as np

DEG = np.pi / 180.0

# [abduction, flexion, elbow] in degrees, segment move time, hold time after
REHAB_TASK = [
    # name,                q_deg,                 move_s, hold_s
    ("rest",               [10.0, 10.0, 20.0],    0.0,    0.5),
    ("forward_reach",      [15.0, 60.0, 30.0],    2.0,    0.5),
    ("hand_to_mouth",      [20.0, 55.0, 125.0],   2.0,    0.5),
    ("lateral_raise",      [75.0, 20.0, 40.0],    2.0,    0.5),
    ("rest",               [10.0, 10.0, 20.0],    2.0,    0.5),
]


def min_jerk(q0, qf, T, t):
    """Quintic minimum-jerk profile. Returns (q, qd, qdd) at times t (any shape)."""
    q0, qf = np.asarray(q0, float), np.asarray(qf, float)
    tau = np.clip(np.asarray(t, float) / T, 0.0, 1.0)[..., None]
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    dq = qf - q0
    return q0 + s * dq, sd * dq, sdd * dq


class WaypointTrajectory:
    def __init__(self, task=REHAB_TASK):
        self.names = [w[0] for w in task]
        self.q = np.array([w[1] for w in task], float) * DEG
        self.move = np.array([w[2] for w in task], float)
        self.hold = np.array([w[3] for w in task], float)
        # segment start times
        self.t_start = []
        t = 0.0
        for i in range(len(task)):
            self.t_start.append(t)
            t += self.move[i] + self.hold[i]
        self.t_start = np.array(self.t_start)
        self.duration = t

    def __call__(self, t):
        """Evaluate (q, qd, qdd) at scalar or array time t (clamped at the end)."""
        t = np.asarray(t, float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        q = np.empty(t.shape + (3,)); qd = np.zeros_like(q); qdd = np.zeros_like(q)
        q[:] = self.q[0]
        for i in range(1, len(self.q)):
            ts = self.t_start[i]
            m = t >= ts
            if not np.any(m):
                continue
            a, b, c = min_jerk(self.q[i - 1], self.q[i], self.move[i], t[m] - ts)
            q[m], qd[m], qdd[m] = a, b, c
        if scalar:
            return q[0], qd[0], qdd[0]
        return q, qd, qdd
