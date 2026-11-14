function L = design_lqr(Ts)
%DESIGN_LQR Integral-augmented LQR (LQI) for the gravity-compensated arm.
%   Linearisation: with tau = G(q) + tau_fb the error dynamics about the
%   mean task inertia M0 are  M0*e_dd + B*e_d = tau_fb  (+ integral state).
%   Weights by Bryson's rule; discrete design (c2d + dlqr) at the 1 kHz
%   control rate. Results -> results/simulink/lqr.mat
if nargin < 1, Ts = 1e-3; end
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here, 'generated'));
outDir = fullfile(here, '..', 'results', 'simulink'); if ~exist(outDir, 'dir'), mkdir(outDir); end
P = load_arm_params();

% mean inertia over the rehab task
t = linspace(0, 10.5, 200); M0 = zeros(3);
for k = 1:numel(t), M0 = M0 + arm_M(rehab_task(t(k)), P.vec); end
M0 = M0 / numel(t);

Mi = inv(M0);
A = [zeros(3) eye(3) zeros(3); zeros(3) -Mi*diag(P.B) zeros(3); eye(3) zeros(3) zeros(3)];
B = [zeros(3); Mi; zeros(3)];
e_max = deg2rad(2)*ones(3,1); ev_max = deg2rad(20)*ones(3,1); z_max = deg2rad(0.5)*ones(3,1);
Q = diag([1./e_max.^2; 1./ev_max.^2; 1./z_max.^2]);
R = diag(1./(0.5*P.tau_max).^2);
sysd = c2d(ss(A, B, eye(9), 0), Ts, 'zoh');
[K, ~, poles_d] = dlqr(sysd.A, sysd.B, Q*Ts, R*Ts);
% u = -K x, x = [q-qr; qd-qrd; int(q-qr)]  ->  tau_fb = Kp e + Kd e_d + Ki z
L.Kp = K(:, 1:3); L.Kd = K(:, 4:6); L.Ki = K(:, 7:9);
L.K = K; L.M0 = M0; L.Q = Q; L.R = R; L.Ts = Ts;
L.poles_equiv_rad_s = log(poles_d)/Ts;
save(fullfile(outDir, 'lqr.mat'), '-struct', 'L');
fprintf('LQR: diag Kp = [%s], diag Ki = [%s], diag Kd = [%s]\n', num2str(diag(L.Kp)', '%.1f '), ...
    num2str(diag(L.Ki)', '%.1f '), num2str(diag(L.Kd)', '%.2f '));
end
