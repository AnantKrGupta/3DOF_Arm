function [out, R] = run_sim(mdl, S)
%RUN_SIM Simulate rehab_arm_3dof with the variables in struct S.
%   R contains N-by-3 arrays t, q, qr, tau, e (rad / Nm).
in = Simulink.SimulationInput(mdl);
f = fieldnames(S);
for k = 1:numel(f)
    in = in.setVariable(f{k}, S.(f{k}));
end
out = sim(in);
R.t   = squeeze(out.get('t_log'));
R.q   = as_rows(out.get('q_log'));
R.qr  = as_rows(out.get('qr_log'));
R.tau = as_rows(out.get('tau_log'));
R.e   = R.qr - R.q;
if ~isempty(out.ErrorMessage), warning('run_sim:error', '%s', out.ErrorMessage); end
end

function A = as_rows(v)
v = squeeze(v);
if size(v, 2) ~= 3, v = v.'; end
A = v;
end
