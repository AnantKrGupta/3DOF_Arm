# 3-DOF Rehab Arm — Kinematics, Control Study & Isaac Sim Digital Twin

Simulation, dynamics/control benchmarking, and digital-twin safety study of a **3-DOF anthropomorphic rehabilitation/assistive arm** matching a left upper-limb BCI prosthesis (`shoulder_abduction`, `shoulder_flexion`, `elbow_flexion`).

A single configuration file ([config/arm_params.json](config/arm_params.json)) drives all four layers — Python kinematics/dynamics, MATLAB/Simulink control, URDF/Simscape Multibody validation, and the OpenUSD/Isaac Sim digital twin.

---

## Highlights & Verified Results

| Study | Key Outcome | Artifacts |
|---|---|---|
| **DH Kinematics & Singularity** | Max reach `0.650 m`; convex-hull workspace `0.962 m³` (`0.563 m³` at 2 cm voxel grid); DLS cuts near-singularity joint rate from `901.2 °/s` (pseudo-inverse) to `91.5 °/s` (`31.4 mm` path deviation). | [results/kinematics/](results/kinematics/) |
| **Euler–Lagrange & Simscape** | Symbolic Euler–Lagrange `M(q), C(q,qd), G(q)` via SymPy + Christoffel symbols; cross-validated against an independent URDF import in Simscape Multibody over a 1.5 s passive swing (`max |err| = [0.017, 0.062, 0.080]°`). | [results/simulink/simscape_validation.png](results/simulink/simscape_validation.png) |
| **GA-PID vs LQR (Simulink)** | 1 kHz gravity-compensated control on the full nonlinear Lagrangian plant across 4 scenarios (zero saturation). LQR achieves `0.020–0.033°` nominal RMSE (`0.067–0.139°` at 3× payload `1.5 kg`, `0.891°` max spasm error) vs GA-PID `0.050–0.064°` nominal (`0.151–0.267°` at `1.5 kg`, `2.099°` max spasm error) with lower torque-rate chatter (`50–77 Nm/s` vs `146–333 Nm/s`). | [results/simulink/metrics_table.md](results/simulink/metrics_table.md) |
| **USD Digital Twin & Safety** | Exported USD articulation (`4` rigid bodies, `3` revolute joints, `4.250 kg`) matches DH FK to `1.19 × 10⁻⁵ mm`. Local 26 s ISO/TS 15066 speed-and-separation preview verifies `RUN` (`17.05 s`), `SLOW` (`3.33 s`), `STOP` (`5.62 s`, `2` stops, hysteresis resume at `0.15 m`). | [isaac/outputs/safety/](isaac/outputs/safety/) |
| **Cloud Isaac Sim + Replicator** | Free **Kaggle T4 GPU** notebook installs headless Isaac Sim, runs the PhysX digital twin (`safety_twin.py` with redundant PhysX overlap-sphere channel & MP4 rendering), and generates randomized RGB/depth/segmentation/2D-box synthetic data (`replicator_sdg.py`). | [isaac/notebooks/isaac_sim_kaggle.ipynb](isaac/notebooks/isaac_sim_kaggle.ipynb) |

---

## Repository Layout

```text
rehab_arm_3dof/
├── config/
│   └── arm_params.json              # Single source of truth: link geometry, inertias, limits, safety thresholds
├── python/
│   ├── rehab_arm/
│   │   ├── params.py                # Parameter loader + DYN_PARAM_NAMES
│   │   ├── kinematics.py            # Standard DH FK, 6x3 Jacobian, 4-branch analytic IK, capsule distances
│   │   ├── trajectory.py            # Quintic minimum-jerk rehab task (rest -> reach -> mouth -> lateral -> rest)
│   │   ├── dynamics_derive.py       # SymPy Euler-Lagrange derivation + Python/MATLAB code generator
│   │   ├── _dynamics_generated.py   # Auto-generated vectorised NumPy M, C, G, energy
│   │   ├── dynamics.py              # ArmDynamics wrapper (forward/inverse dynamics)
│   │   ├── control.py               # Batched 1 kHz simulator, real-coded GA, discrete LQI
│   │   └── assets.py                # URDF + USD Physics articulation exporter & USD FK reader
│   └── scripts/
│       ├── 01_kinematics_study.py   # Workspace, singularity loci, SVD, pinv vs DLS resolved-rate study
│       ├── 02_derive_dynamics.py    # Generates _dynamics_generated.py and matlab/generated/*.m
│       ├── 03_control_study.py      # Python mirror of GA-PID vs LQR study
│       ├── 04_export_assets.py      # Exports URDF + USD articulation and checks stage properties
│       └── 05_local_twin_preview.py # Local USD FK verification + 26 s ISO/TS 15066 safety preview
├── matlab/
│   ├── generated/                   # SymPy-generated codegen-ready arm_M.m, arm_C.m, arm_G.m, arm_energy.m
│   ├── build_rehab_arm_model.m      # Programmatically builds rehab_arm_3dof.slx
│   ├── design_lqr.m                 # ZOH-discretised LQI controller (c2d + dlqr)
│   ├── tune_pid_ga.m                # Global Optimization Toolbox ga() over 9 PID gains with Fast Restart
│   ├── compare_controllers.m        # Runs 4 robustness scenarios and exports plots/metrics
│   ├── validate_simscape.m          # Imports URDF via smimport and cross-validates passive swing
│   ├── run_all.m                    # One-click MATLAB/Simulink entry point
│   └── rehab_arm_3dof.slx           # Generated Simulink plant + controller model
├── isaac/
│   ├── assets/                      # Generated rehab_arm.urdf & rehab_arm.usda
│   ├── scripts/
│   │   ├── common.py                # Clinic scene builder, semantics, Isaac Sim 4.x/5.x compat layer
│   │   ├── safety_twin.py           # Isaac Sim USD import, gravity comp check, 2-channel safety + MP4
│   │   ├── replicator_sdg.py        # Domain-randomised Replicator synthetic dataset generator
│   │   └── plot_twin_results.py     # Plots twin_log.csv -> safety_timeline.png
│   ├── notebooks/
│   │   ├── make_notebook.py         # Generator for the Kaggle/Colab notebook
│   │   └── isaac_sim_kaggle.ipynb   # Ready-to-run free T4 GPU notebook
│   └── outputs/safety/              # Verified local USD FK & 26 s safety preview plots/logs
├── tests/
│   └── test_model.py                # 10 self-checks (FK, IK, Jacobian, SPD/skew M-2C, energy, USD FK, LQI)
├── report/
│   └── REPORT.md                    # Full technical report with equations, tables, and analysis
├── requirements.txt
└── README.md
```

---

## Quickstart (Local)

### 1. Python setup & unit tests
```bash
pip install -r requirements.txt
python tests/test_model.py
```

### 2. Kinematics, Lagrangian derivation, assets & Python control mirror
```bash
python python/scripts/01_kinematics_study.py
python python/scripts/02_derive_dynamics.py
python python/scripts/04_export_assets.py
python python/scripts/05_local_twin_preview.py
python python/scripts/03_control_study.py      # ~9 min for full 40x30 GA (use --quick for smoke test)
```

### 3. MATLAB / Simulink (R2025a + Simulink + Control System + Global Optimization + Simscape Multibody)
```matlab
cd matlab
run_all
```
This builds `rehab_arm_3dof.slx`, computes LQI gains (`design_lqr`), tunes PID via `ga()` (`tune_pid_ga(40,30)`), benchmarks all 4 scenarios (`compare_controllers`), and validates the Lagrangian equations against Simscape Multibody (`validate_simscape`).

---

## Running Isaac Sim & Replicator on Free Cloud GPU (Kaggle)

Since Isaac Sim requires an NVIDIA RTX/T4 GPU with Vulkan + OptiX drivers, [isaac/notebooks/isaac_sim_kaggle.ipynb](isaac/notebooks/isaac_sim_kaggle.ipynb) runs headless Isaac Sim for free on **Kaggle Notebooks** (T4 GPU, 30 hrs/week, ~29 GB RAM):

1. Push this repository to a public GitHub repo (e.g. `https://github.com/<your-username>/rehab-arm-3dof.git`).
2. Create a [Kaggle](https://www.kaggle.com/) account and **verify your phone number** under *Account Settings* (required to enable GPU and Internet).
3. Import `isaac/notebooks/isaac_sim_kaggle.ipynb` into a new Kaggle Notebook.
4. In the right-hand **Notebook Settings** panel:
   - Set **Accelerator** to `GPU T4 x1` (or `T4 x2`).
   - Toggle **Internet** to `On`.
5. In the first code cell, update `REPO_URL` to your GitHub URL and click **Run All**.
6. Wait ~20–40 min on first run (extracts Vulkan/OptiX user-space libs, installs `isaacsim[all,extscache]==5.1.0` into an isolated Python 3.11 `uv` venv, compiles RTX shaders, runs `safety_twin.py` and `replicator_sdg.py`).
7. Download `/kaggle/working/twin_outputs.zip` from the **Output** tab (`safety_demo.mp4`, `twin_log.csv`, `validation.json`, `summary.json`, and 300 Replicator RGB/depth/segmentation/bbox frames + `labels.csv`).

---

## Technical Report

See [report/REPORT.md](report/REPORT.md) for full derivations, DH table, singularity analysis, Simscape validation plots, controller comparison tables, and digital-twin architecture.
