"""Self-checks for kinematics and the generated Lagrangian dynamics.

Run:  python -m pytest tests -q      (or: python tests/test_model.py)
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from rehab_arm import ROOT, ThreeDofArm, WaypointTrajectory, load_params  # noqa: E402
from rehab_arm.assets import ASSET_DIR, UsdArticulationFK  # noqa: E402
from rehab_arm.control import Gains, Scenario, lqi_gains, mean_task_inertia, metrics, simulate  # noqa: E402
from rehab_arm.dynamics import ArmDynamics  # noqa: E402

RNG = np.random.default_rng(0)
ARM = ThreeDofArm()
DYN = ArmDynamics()


def test_fk_closed_form_matches_dh():
    Q = ARM.sample_joint_space(200, RNG)
    P = ARM.fk_batch(Q, world=True)
    for q, p in zip(Q, P):
        assert np.allclose(ARM.fk(q, world=True), p, atol=1e-12)


def test_neutral_pose_hangs_down():
    p = ARM.fk([0, 0, 0], world=True)
    assert np.allclose(p, ARM.p.shoulder_pos + [0, 0, -(ARM.L1 + ARM.L2)])
    # flexion moves hand forward (+X), abduction moves left (+Y)
    assert ARM.fk([0, np.pi / 2, 0], world=True)[0] > 0.6
    assert ARM.fk([np.pi / 2, 0, 0], world=True)[1] > 0.6


def test_ik_roundtrip():
    Q = ARM.sample_joint_space(300, RNG)
    for q in Q:
        sols = ARM.ik(ARM.fk(q, world=True), world=True, within_limits=False)
        assert any(np.allclose(ARM.fk(s, world=True), ARM.fk(q, world=True), atol=1e-9) for s in sols)


def test_jacobian_vs_finite_difference_and_det():
    for q in ARM.sample_joint_space(50, RNG):
        J = ARM.jacobian_pos(q)
        eps = 1e-7
        Jfd = np.column_stack([(ARM.fk(q + eps * e) - ARM.fk(q - eps * e)) / (2 * eps) for e in np.eye(3)])
        assert np.allclose(J, Jfd, atol=1e-6)
        assert np.isclose(np.linalg.det(J), ARM.det_jv(q), atol=1e-9)


def test_mass_matrix_spd_and_skew_property():
    for _ in range(50):
        q = ARM.sample_joint_space(1, RNG)[0]
        qd = RNG.normal(size=3)
        M = DYN.M(q)
        assert np.allclose(M, M.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(M) > 0)
        eps = 1e-7
        Mdot = (DYN.M(q + eps * qd) - DYN.M(q - eps * qd)) / (2 * eps)
        S = Mdot - 2 * DYN.C(q, qd)
        assert np.allclose(S, -S.T, atol=1e-6)


def test_gravity_is_gradient_of_potential():
    for q in ARM.sample_joint_space(30, RNG):
        eps = 1e-6
        dV = [(DYN.energy(q + eps * e, np.zeros(3))[1] - DYN.energy(q - eps * e, np.zeros(3))[1]) / (2 * eps)
              for e in np.eye(3)]
        assert np.allclose(DYN.G(q), dV, atol=1e-6)


def test_static_shoulder_torque_hand_calc():
    # arm horizontal forward (q2 = 90 deg, elbow straight): shoulder flexion torque
    p = load_params()
    tau = DYN.G(np.array([0, np.pi / 2, 0]))
    expected = p.g * (p.m1 * p.c1 * p.L1 + p.m2 * (p.L1 + p.c2 * p.L2) + p.mp * (p.L1 + p.L2))
    assert np.isclose(abs(tau[1]), expected, rtol=1e-9)


def test_energy_conservation_passive():
    dyn = ArmDynamics()
    dyn.B[:] = 0.0
    q = np.array([0.4, 0.8, 0.6]); qd = np.zeros(3)
    E0 = sum(dyn.energy(q, qd))
    dt = 2e-4
    for _ in range(int(2.0 / dt)):  # RK4, 2 s of free swinging
        def f(x):
            return np.concatenate([x[3:], dyn.forward(x[:3], x[3:], np.zeros(3))])
        x = np.concatenate([q, qd])
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q, qd = x[:3], x[3:]
    E1 = sum(dyn.energy(q, qd))
    assert abs(E1 - E0) < 1e-5 * max(1.0, abs(E0))


def test_usd_asset_and_fk():
    from pxr import Usd, UsdPhysics

    usd_path = ASSET_DIR / "rehab_arm.usda"
    assert usd_path.exists(), f"Missing USD asset: {usd_path}"
    st = Usd.Stage.Open(str(usd_path))
    bodies = [p for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in st.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert len(bodies) == 4
    assert len(joints) == 3
    total_mass = sum(UsdPhysics.MassAPI(b).GetMassAttr().Get() for b in bodies)
    assert np.isclose(total_mass, 4.250, atol=1e-5)

    usd_fk = UsdArticulationFK(usd_path, ARM.p)
    for q in ARM.sample_joint_space(50, RNG):
        assert np.allclose(usd_fk.ee_world(q), ARM.fk(q, world=True), atol=1e-5)


def test_lqi_stability_and_tracking():
    traj = WaypointTrajectory()
    M0 = mean_task_inertia(DYN, traj)
    deg = np.pi / 180.0
    Kp, Ki, Kd, _, poles = lqi_gains(
        DYN, M0, [2 * deg] * 3, [20 * deg] * 3, [0.5 * deg] * 3, 0.5 * ARM.p.tau_max, dt=1e-3
    )
    assert np.all(poles.real < 0)
    res = simulate(Gains.from_matrices(Kp, Ki, Kd), Scenario("nominal", seed=1), traj, model=DYN)
    m = metrics(res, 0)
    assert not m["diverged"]
    assert np.all(np.array(m["rmse_deg"]) < 0.1)


if __name__ == "__main__":
    _ = ROOT
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
