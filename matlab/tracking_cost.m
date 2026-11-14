function [J, m] = tracking_cost(R, P)
%TRACKING_COST GA fitness (identical weights to python/rehab_arm/control.py)
%   J = 1e4*ISE + 1e-3*int(tau^2) + 2e-6*int(tau_dot^2) + 50*t_saturated
dt   = R.t(2) - R.t(1);
ise  = sum(sum(R.e.^2, 2)) * dt;
eff  = sum(sum(R.tau.^2, 2)) * dt;
dtau = sum(sum((diff(R.tau) / dt).^2, 2)) * dt;
sat  = sum(any(abs(R.tau) >= P.tau_max.' - 1e-9, 2)) * dt;
J = 1e4*ise + 1e-3*eff + 2e-6*dtau + 50*sat;
if any(~isfinite(R.q(:))) || any(abs(R.q(:)) > 10), J = 1e6; end
ed = rad2deg(R.e);
m = struct('rmse_deg', sqrt(mean(ed.^2, 1)), 'max_err_deg', max(abs(ed), [], 1), ...
    'rms_tau_Nm', sqrt(mean(R.tau.^2, 1)), 'peak_tau_Nm', max(abs(R.tau), [], 1), ...
    'tau_rate_rms_Nm_s', sqrt(mean((diff(R.tau)/dt).^2, 1)), ...
    'saturation_time_s', sat, 'ISE_rad2s', ise, 'cost', J);
end
