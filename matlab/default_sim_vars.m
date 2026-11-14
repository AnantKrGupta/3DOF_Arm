function S = default_sim_vars(P)
%DEFAULT_SIM_VARS Variables consumed by rehab_arm_3dof.slx (nominal scenario).
if nargin < 1, P = load_arm_params(); end
S.Ts       = 1e-3;
S.Tsim     = 10.5;
S.Pm       = P.vec;            % model parameters (gravity compensation)
S.Pt       = P.vec;            % TRUE plant parameters
S.Bv       = P.B;
S.tau_max  = P.tau_max;
S.q0       = rehab_task(0);
S.noise_q  = 2e-4;             % rad
S.noise_qd = 5e-3;             % rad/s
S.dist     = [0 0 1 0; 0 0 1 0];   % [t0 t1 joint Nm] (two rows, fixed size)
S.Kp = zeros(3); S.Ki = zeros(3); S.Kd = zeros(3);
end
