# Technical Report: 3-DOF Rehab Arm — Kinematics, Control Study & Digital Twin

## 1. System Overview & Parameterization

The modelled device is a 3-DOF anthropomorphic rehabilitation/assistive robotic arm configured to match a **left human upper limb** (`config/arm_params.json`):

| Parameter | Symbol | Value |
|---|---|---|
| Upper arm length / radius / mass / COM ratio | $L_1, r_1, m_1, c_1$ | `0.30 m`, `0.035 m`, `1.60 kg`, `0.45` |
| Forearm + hand length / radius / mass / COM ratio | $L_2, r_2, m_2, c_2$ | `0.35 m`, `0.030 m`, `1.10 kg`, `0.42` |
| End-effector payload mass | $m_p$ | `0.50 kg` (nominal), `1.50 kg` (heavy test) |
| Joint 1: `shoulder_abduction` limits / $\dot{q}_{\max}$ / $\tau_{\max}$ / $B_1$ | $q_1$ | `[-20, 150]°`, `90 °/s`, `25 Nm`, `0.15 Nms/rad` |
| Joint 2: `shoulder_flexion` limits / $\dot{q}_{\max}$ / $\tau_{\max}$ / $B_2$ | $q_2$ | `[-45, 160]°`, `90 °/s`, `25 Nm`, `0.15 Nms/rad` |
| Joint 3: `elbow_flexion` limits / $\dot{q}_{\max}$ / $\tau_{\max}$ / $B_3$ | $q_3$ | `[0, 145]°`, `120 °/s`, `12 Nm`, `0.08 Nms/rad` |
| World shoulder origin | $\mathbf{p}_{\text{shoulder}}$ | `[0.0, 0.0, 1.20] m` |

The benchmark rehabilitation task (`python/rehab_arm/trajectory.py` and `matlab/rehab_task.m`) links five joint-space postures using quintic minimum-jerk profiles (`2.0 s` transitions, `0.5 s` holds, `10.5 s` total):
`rest [10, 10, 20]°` $\rightarrow$ `forward_reach [15, 60, 30]°` $\rightarrow$ `hand_to_mouth [20, 55, 125]°` $\rightarrow$ `lateral_raise [75, 20, 40]°` $\rightarrow$ `rest [10, 10, 20]°`.

---

## 2. Denavit–Hartenberg Kinematics, Workspace & Singularity Analysis

### 2.1 Standard DH Table & Frame Mounting

| $i$ | $a_i$ [m] | $\alpha_i$ [deg] | $d_i$ [m] | $\theta_i$ | Joint |
|---|---|---|---|---|---|
| 1 | 0.000 | +90 | 0 | $q_1$ | `shoulder_abduction` |
| 2 | 0.300 | 0 | 0 | $q_2$ | `shoulder_flexion` |
| 3 | 0.350 | 0 | 0 | $q_3$ | `elbow_flexion` |

The DH base frame $\{0\}$ is mounted at $\mathbf{p}_{\text{shoulder}}$ with $\hat{\mathbf{z}}_0 = +\hat{\mathbf{X}}_{\text{world}}$ (anterior), $\hat{\mathbf{x}}_0 = -\hat{\mathbf{Z}}_{\text{world}}$ (downward), and $\hat{\mathbf{y}}_0 = +\hat{\mathbf{Y}}_{\text{world}}$ (left). With $A = L_1 \cos q_2 + L_2 \cos(q_2 + q_3)$, the closed-form end-effector position in $\{0\}$ and translational Jacobian determinant are:

$$\mathbf{p}_0(\mathbf{q}) = \begin{bmatrix} A \cos q_1 \\ A \sin q_1 \\ L_1 \sin q_2 + L_2 \sin(q_2 + q_3) \end{bmatrix}, \qquad \det \mathbf{J}_v(\mathbf{q}) = -L_1 L_2 \sin q_3 \bigl(L_1 \cos q_2 + L_2 \cos(q_2 + q_3)\bigr).$$

An analytic 4-branch inverse kinematics solver (`ThreeDofArm.ik`) enumerates both shoulder ($A > 0 / A < 0$) and elbow-up/down ($\sin q_3 > 0 / < 0$) configurations and filters by joint limits.

### 2.2 Workspace & Singularity Results (`results/kinematics/`)

Across `200,000` uniform joint-space samples (`kinematics_summary.json`):
- **Max reach**: `0.650 m`; **Bounding box**: $X \in [-0.454, 0.650]$ m, $Y \in [-0.631, 0.650]$ m, $Z \in [0.550, 1.831]$ m.
- **Workspace volume**: `0.9616 m³` (convex hull) and `0.5632 m³` (2 cm voxel occupancy).
- **Yoshikawa manipulability** $w(\mathbf{q}) = |\det \mathbf{J}_v|$: mean `0.0235 m³`, max `0.0526 m³`; **13.5%** of valid configurations lie near a singularity ($w < 0.1\, w_{\max}$).
- **Analytic singular loci verified by SVD**:
  1. *Elbow boundary singularity* ($\sin q_3 = 0$, tested at $q = [17.19, 34.38, 0.0]^\circ$): $\mathbf{s} = [0.738241, 0.536468, 0.0]$, losing radial extension motion.
  2. *Shoulder internal singularity* ($A = 0$, hand on the anterior $q_1$ abduction axis, tested at $q = [28.65, 58.95, 57.30]^\circ$): $\mathbf{s} = [0.655990, 0.134689, 0.0]$, losing motion perpendicular to the arm plane.
- **Resolved-rate control 2 cm from shoulder singularity** (`singularity_resolved_rate.png`):
  - **Moore–Penrose pseudo-inverse** (`pinv`): tracks to `0.11 mm` max error, but spikes joint 1 rate to **`901.2 °/s`** ($10\times$ above $\dot{q}_{1,\max} = 90\text{ °/s}$).
  - **Damped Least Squares** (`dls`, $w_0 = 0.03$, $\lambda_{\max} = 0.15$): limits peak joint 1 rate to **`91.5 °/s`** while bounding transient hand path error to `31.4 mm`.
  - Because the `forward_reach` $\rightarrow$ `hand_to_mouth` $\rightarrow$ `lateral_raise` transitions cross $A = 0$ (minimum task manipulability `1.91e-4 m³`), joint-space minimum-jerk planning is used for control execution.

| Workspace 3D | Singularity Map | Resolved-Rate (pinv vs DLS) |
|---|---|---|
| ![Workspace](file:///D:/Mitacs_projects/rehab_arm_3dof/results/kinematics/workspace_3d.png) | ![Singularity Map](file:///D:/Mitacs_projects/rehab_arm_3dof/results/kinematics/singularity_map.png) | ![Resolved Rate](file:///D:/Mitacs_projects/rehab_arm_3dof/results/kinematics/singularity_resolved_rate.png) |

---

## 3. Euler–Lagrange Dynamics & Simscape Cross-Validation

The rigid-body equations of motion were derived symbolically in SymPy (`python/rehab_arm/dynamics_derive.py`) modelling the upper arm and forearm as solid cylinders of radii $r_1, r_2$ and the payload $m_p$ as a distal point mass:

$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) + \mathbf{B}\dot{\mathbf{q}} = \boldsymbol{\tau}$$

where $\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})$ is constructed from Christoffel symbols of the first kind so that $\dot{\mathbf{M}} - 2\mathbf{C}$ is skew-symmetric. Code generation produces vectorised NumPy (`_dynamics_generated.py`) and Simulink-compatible MATLAB (`matlab/generated/arm_{M,C,G,energy}.m`) functions.

### Verification
- **Analytical & Numerical Self-Checks** (`tests/test_model.py`, all 10 tests pass): $\mathbf{M} \succ 0$, $\dot{\mathbf{M}} - 2\mathbf{C}$ skew-symmetric to $10^{-6}$, $\mathbf{G} = \nabla V$ to $10^{-6}$, static horizontal torque matches hand calculation to $10^{-9}$ relative tolerance, and RK4 passive energy drift $< 10^{-5}$ over 2 s.
- **Simscape Multibody Cross-Validation** (`matlab/validate_simscape.m`): Importing `isaac/assets/rehab_arm.urdf` via `smimport` yields an independent multibody model. Over a `1.5 s` passive gravity swing from $\mathbf{q}_0 = [30, 60, 45]^\circ$:
  - **Max absolute error**: `[0.0168, 0.0618, 0.0799]°`
  - **RMS error**: `[0.0078, 0.0263, 0.0261]°`
  (Small residual comes from the `0.05 kg` intermediate shoulder housing mass and `9.80665` vs `9.81 m/s²` Simscape default gravity.)

![Simscape Validation](file:///D:/Mitacs_projects/rehab_arm_3dof/results/simulink/simscape_validation.png)

---

## 4. Control Study: GA-Tuned PID vs LQR (LQI)

Both controllers share an identical 1 kHz gravity-compensated control architecture with actuator saturation and conditional-integration anti-windup:

$$\boldsymbol{\tau}_{\text{u}} = \hat{\mathbf{G}}(\mathbf{q}_m) + \mathbf{K}_p \mathbf{e} + \mathbf{K}_i \int \mathbf{e}\,dt + \mathbf{K}_d \dot{\mathbf{e}}, \qquad \mathbf{e} = \mathbf{q}_{\text{ref}} - \mathbf{q}_m$$

- **GA-PID**: Diagonal $\mathbf{K}_p, \mathbf{K}_i, \mathbf{K}_d$ (`9` decision variables). Tuned with MATLAB `ga()` (`pop = 40, gens = 30`, Fast Restart) and mirrored in Python (`pop = 40, gens = 30`) minimizing:
  $$J = 10^4 \!\int \|\mathbf{e}\|^2 dt + 10^{-3} \!\int \|\boldsymbol{\tau}\|^2 dt + 2\times 10^{-6} \!\int \|\dot{\boldsymbol{\tau}}\|^2 dt + 50\,t_{\text{sat}}.$$
  Bounds enforce derivative damping and bounded integral action (`Kp ∈ [50,800]/[25,400]`, `Ki ∈ [20,800]/[10,400]`, `Kd ∈ [8,50]/[4,25]`) so the Routh–Hurwitz stability margin $(B + K_d)K_p > M K_i$ holds under heavy payload variations.
  - **MATLAB GA gains** (`J = 1.0813`, 1182 sims): `Kp = [148.88, 155.91, 113.66]`, `Ki = [653.11, 759.60, 398.07]`, `Kd = [8.48, 8.47, 4.59]`.
  - **Python GA gains** (`J = 1.0349`): `Kp = [134.38, 180.76, 152.15]`, `Ki = [780.69, 750.41, 349.68]`, `Kd = [8.00, 8.00, 4.01]`.
- **LQR (LQI)**: Full `3×3` coupled gain matrices computed via ZOH-discretised `c2d + dlqr` (`Ts = 1 ms`) on the error dynamics linearized at the task-mean inertia $\mathbf{M}_0$:
  $$\mathbf{M}_0 = \begin{bmatrix} 0.2022 & 0 & 0 \\ 0 & 0.3897 & 0.1503 \\ 0 & 0.1503 & 0.0965 \end{bmatrix}\text{ kg m}^2$$
  with Bryson's rule weights ($e_{\max} = 2^\circ$, $\dot{e}_{\max} = 20^\circ\text{/s}$, $z_{\max} = 0.5^\circ$, $\tau_{\text{fb,max}} = 0.5\,\boldsymbol{\tau}_{\max}$), yielding diagonal gains `diag(Kp) = [444.9, 427.1, 181.3]`, `diag(Ki) = [1303.1, 1230.3, 529.8]`, `diag(Kd) = [35.08, 35.56, 14.40]` plus inertial decoupling cross-terms between joints 2 and 3.

### Benchmark Results (Simulink `rehab_arm_3dof.slx`)

| Scenario | Controller | RMSE [deg] `(j1, j2, j3)` | Max Err [deg] | Peak $\tau$ [Nm] | $\dot{\tau}$ RMS [Nm/s] | Sat. [s] |
|---|---|---|---|---|---|---|
| `nominal` | **GA-PID** | `[0.050, 0.064, 0.050]` | `0.171` | `[7.96, 9.43, 3.51]` | `[77, 76, 50]` | `0` |
| `nominal` | **LQR** | `[0.020, 0.033, 0.023]` | `0.083` | `[8.49, 9.81, 3.70]` | `[333, 308, 146]` | `0` |
| `heavy_payload_1.5kg` | **GA-PID** | `[0.151, 0.219, 0.267]` | `0.908` | `[12.31, 15.80, 7.18]` | `[76, 76, 49]` | `0` |
| `heavy_payload_1.5kg` | **LQR** | `[0.067, 0.106, 0.139]` | `0.386` | `[12.79, 16.02, 7.27]` | `[318, 303, 139]` | `0` |
| `mass+20%_noise_x3` | **GA-PID** | `[0.050, 0.066, 0.060]` | `0.201` | `[9.41, 10.97, 3.99]` | `[229, 228, 148]` | `0` |
| `mass+20%_noise_x3` | **LQR** | `[0.020, 0.034, 0.031]` | `0.089` | `[11.00, 12.37, 4.56]` | `[983, 913, 433]` | `0` |
| `spasm_disturbance` | **GA-PID** | `[0.050, 0.347, 0.291]` | `2.099` | `[8.11, 10.72, 3.51]` | `[77, 78, 52]` | `0` |
| `spasm_disturbance` | **LQR** | `[0.020, 0.107, 0.162]` | `0.891` | `[8.47, 12.14, 3.70]` | `[333, 310, 147]` | `0` |

### Python Cross-Check (`results/control_python/metrics_table.md`)

| Scenario | Controller | RMSE [deg] `(j1, j2, j3)` | Max Err [deg] | Peak $\tau$ [Nm] | $\dot{\tau}$ RMS [Nm/s] |
|---|---|---|---|---|---|
| `nominal` | **GA-PID** | `[0.047, 0.060, 0.045]` | `0.163` | `[7.95, 9.41, 3.56]` | `[70.5, 78.5, 54.8]` |
| `nominal` | **LQR** | `[0.020, 0.033, 0.023]` | `0.085` | `[8.36, 9.83, 3.70]` | `[381.7, 325.0, 159.9]` |
| `heavy_payload_1.5kg` | **GA-PID** | `[0.139, 0.194, 0.250]` | `0.848` | `[12.28, 15.74, 7.18]` | `[71.0, 77.7, 54.1]` |
| `heavy_payload_1.5kg` | **LQR** | `[0.068, 0.106, 0.139]` | `0.382` | `[12.64, 15.98, 7.28]` | `[339.3, 309.0, 142.5]` |
| `mass+20%_noise_x3` | **GA-PID** | `[0.049, 0.061, 0.055]` | `0.184` | `[9.34, 10.93, 4.05]` | `[212.8, 230.8, 165.1]` |
| `mass+20%_noise_x3` | **LQR** | `[0.022, 0.034, 0.032]` | `0.085` | `[10.96, 12.42, 4.60]` | `[1053.3, 953.1, 448.5]` |
| `spasm_disturbance` | **GA-PID** | `[0.047, 0.301, 0.227]` | `1.814` | `[8.07, 10.85, 3.56]` | `[70.9, 79.0, 57.0]` |
| `spasm_disturbance` | **LQR** | `[0.021, 0.107, 0.163]` | `0.890` | `[8.39, 12.18, 3.73]` | `[373.7, 324.9, 160.3]` |

**Key Tradeoff**: LQR's higher bandwidth and explicit inertial decoupling between `shoulder_flexion` and `elbow_flexion` halve tracking error across all scenarios and suppress the `1.5 kg` unmodelled payload (`0.067–0.139°` RMSE) and spasm torque pulses (`0.891°` max error). GA-PID trades ~2× higher tracking error (still `< 0.35°` RMSE) for ~4× lower torque-rate noise amplification (`50–77 Nm/s` nominal, `148–229 Nm/s` under `3×` sensor noise vs `433–983 Nm/s` for LQR).

| Nominal Tracking (Simulink) | Robustness Comparison (Simulink) |
|---|---|
| ![Tracking](file:///D:/Mitacs_projects/rehab_arm_3dof/results/simulink/tracking_nominal.png) | ![Robustness](file:///D:/Mitacs_projects/rehab_arm_3dof/results/simulink/robustness_rmse.png) |

---

## 5. Digital Twin, Proximity Safety & Replicator SDG

### 5.1 USD Articulation Export & Kinematic Verification

`python/rehab_arm/assets.py` exports `isaac/assets/rehab_arm.usda` (with `UsdPhysics.ArticulationRootAPI`, 4 rigid bodies totaling `4.250 kg`, 3 revolute joints with angular drives `Kp = 400 Nm/rad`, `Kd = 30 Nm s/rad`, effort limits, and `PhysicsFilteredPairsAPI` disabling self-collision) alongside `isaac/assets/rehab_arm.urdf`.

Traversing `rehab_arm.usda` directly with `pxr.Usd` across 6 validation poses and 50 random samples yields a **maximum end-effector discrepancy of `1.19 × 10⁻⁵ mm`** against `ThreeDofArm.fk()` (`isaac/outputs/safety/validation.json`, `tests/test_model.py`).

### 5.2 ISO/TS 15066 Speed & Separation Monitoring

The safety monitor time-scales the trajectory phase $s_{\text{task}}$ via $\dot{s}_{\text{task}} = \alpha(d)$ using capsule-to-sphere surface distance $d$:
- **RUN** ($d > 0.30\text{ m}$): $\alpha = 1.0$.
- **SLOW** ($0.10\text{ m} < d \le 0.30\text{ m}$): $\alpha = \text{clip}\bigl((d - 0.10)/(0.30 - 0.10),\,0,\,1\bigr)$, rate-limited (`4 s⁻¹` accel, `12 s⁻¹` decel).
- **PROTECTIVE STOP** ($d \le 0.10\text{ m}$ or redundant PhysX overlap-sphere hit): $\alpha = 0.0$; motion resumes only after hysteresis clearance $d > 0.15\text{ m}$.

Local verification (`python/scripts/05_local_twin_preview.py`, 960 Hz RK4 dynamics with USD drive gains + `dyn.G(q)`) over the 26 s human-hand approach script logged:
- **Time in states**: `RUN = 17.05 s`, `SLOW = 3.33 s`, `STOP = 5.62 s` (`2` protective stops).
- **Minimum separation**: `0.0372 m` (`3.72 cm`, positive clearance maintained).
- **Max twin EE mismatch during motion**: `1.19 × 10⁻⁵ mm`.

| Safety Timeline | 3D Safety Keyframes |
|---|---|
| ![Safety Timeline](file:///D:/Mitacs_projects/rehab_arm_3dof/isaac/outputs/safety/safety_timeline.png) | ![3D Keyframes](file:///D:/Mitacs_projects/rehab_arm_3dof/isaac/outputs/safety/safety_3d_keyframes.png) |

### 5.3 Cloud Execution (Isaac Sim + Replicator SDG)

Because Isaac Sim's RTX renderer and PhysX overlap-sphere queries require an NVIDIA GPU with Vulkan/OptiX drivers, `isaac/notebooks/isaac_sim_kaggle.ipynb` executes the full cloud pipeline on a free Kaggle T4 GPU:
1. `isaac/scripts/safety_twin.py`: Imports `rehab_arm.usda` into a clinic scene, validates gravity compensation hold accuracy, runs both redundant safety channels (analytic capsules + `get_physx_scene_query_interface().overlap_sphere`), and records `safety_demo.mp4`.
2. `isaac/scripts/replicator_sdg.py`: Captures 300 domain-randomized frames (joint angles, obstacle proximity with 50% near-arm bias, hemisphere camera poses, dome/key lighting) outputting RGB, `distance_to_camera` depth, semantic segmentation (7 classes), 2D tight bounding boxes, and `labels.csv` with 3D/2D projected keypoints.
