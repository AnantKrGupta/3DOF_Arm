"""Numeric dynamics wrapper around the SymPy-generated code."""
from __future__ import annotations

import numpy as np

try:
    from . import _dynamics_generated as _gen
except ImportError as e:  # pragma: no cover
    raise ImportError("Run `python python/scripts/02_derive_dynamics.py` first to generate "
                      "rehab_arm/_dynamics_generated.py") from e

from .params import ArmParams, load_params


class ArmDynamics:
    """M(q) qdd + C(q,qd) qd + G(q) + B qd = tau   (all vectorised over (...,3))."""

    def __init__(self, params: ArmParams | None = None, payload: float | None = None):
        self.params = params or load_params()
        self.P = self.params.dyn_vector(payload)
        self.B = self.params.viscous.copy()

    def M(self, q):
        return _gen.mass_matrix(q, self.P)

    def C(self, q, qd):
        return _gen.coriolis_matrix(q, qd, self.P)

    def G(self, q):
        return _gen.gravity_vector(q, self.P)

    def energy(self, q, qd):
        ke_v = _gen.energy(q, qd, self.P)
        return ke_v[..., 0], ke_v[..., 1]

    def forward(self, q, qd, tau):
        """qdd = M^-1 (tau - C qd - G - B qd)."""
        rhs = tau - np.einsum("...ij,...j->...i", self.C(q, qd), qd) - self.G(q) - self.B * qd
        return np.linalg.solve(self.M(q), rhs[..., None])[..., 0]

    def inverse(self, q, qd, qdd):
        return (np.einsum("...ij,...j->...i", self.M(q), qdd)
                + np.einsum("...ij,...j->...i", self.C(q, qd), qd) + self.G(q) + self.B * qd)
