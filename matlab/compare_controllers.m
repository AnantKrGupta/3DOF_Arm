function C = compare_controllers()
%COMPARE_CONTROLLERS GA-tuned PID vs LQR on the Simulink Lagrangian model.
%   Scenarios: nominal, heavy payload (1.5 kg true, 0.5 kg model),
%   +20% link masses with 3x sensor noise, spasm-like torque pulses.
%   Outputs -> results/simulink/{*.png, metrics_table.md, summary.json}
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here, 'generated'));
outDir = fullfile(here, '..', 'results', 'simulink');
mdl = 'rehab_arm_3dof';
load_system(fullfile(here, [mdl '.slx']));
P = load_arm_params();
G = load(fullfile(outDir, 'ga_pid.mat'));
L = load(fullfile(outDir, 'lqr.mat'));
ctrl = struct('name', {'GA-PID', 'LQR'}, 'Kp', {G.Kp, L.Kp}, 'Ki', {G.Ki, L.Ki}, 'Kd', {G.Kd, L.Kd});

S0 = default_sim_vars(P);
sc = struct('name', {}, 'S', {});
sc(1).name = 'nominal';            sc(1).S = S0;
s = S0; s.Pt(7) = 1.5;             sc(2).name = 'heavy_payload_1.5kg'; sc(2).S = s;
s = S0; s.Pt(5:6) = 1.2*s.Pt(5:6); s.noise_q = 3*s.noise_q; s.noise_qd = 3*s.noise_qd;
sc(3).name = 'mass+20%_noise_x3';  sc(3).S = s;
s = S0; s.dist = [3.6 3.9 3 4.0; 7.5 7.8 2 -6.0];
sc(4).name = 'spasm_disturbance';  sc(4).S = s;

C = struct(); R = cell(numel(sc), 2);
lines = {'| scenario | controller | RMSE [deg] (j1,j2,j3) | max err [deg] | peak tau [Nm] | tau-rate RMS [Nm/s] |', ...
         '|---|---|---|---|---|---|'};
for i = 1:numel(sc)
    for c = 1:2
        S = sc(i).S; S.Kp = ctrl(c).Kp; S.Ki = ctrl(c).Ki; S.Kd = ctrl(c).Kd;
        [~, R{i, c}] = run_sim(mdl, S);
        [~, m] = tracking_cost(R{i, c}, P);
        C.(matlab.lang.makeValidName(sc(i).name)).(matlab.lang.makeValidName(ctrl(c).name)) = m;
        lines{end+1} = sprintf('| %s | %s | [%s] | %.3f | [%s] | [%s] |', sc(i).name, ctrl(c).name, ...
            num2str(m.rmse_deg, '%.3f '), max(m.max_err_deg), num2str(m.peak_tau_Nm, '%.2f '), ...
            num2str(m.tau_rate_rms_Nm_s, '%.0f ')); %#ok<AGROW>
        fprintf('%-22s %-7s RMSE = [%s] deg\n', sc(i).name, ctrl(c).name, num2str(m.rmse_deg, '%.3f '));
    end
end
writelines(string(lines), fullfile(outDir, 'metrics_table.md'));
C.ga = struct('Kp', diag(G.Kp)', 'Ki', diag(G.Ki)', 'Kd', diag(G.Kd)', 'J', G.J, 'nEval', G.nEval);
C.lqr = struct('Kp', L.Kp, 'Ki', L.Ki, 'Kd', L.Kd);
fid = fopen(fullfile(outDir, 'summary.json'), 'w'); fprintf(fid, '%s', jsonencode(C, 'PrettyPrint', true)); fclose(fid);

% ------------------------------------------------------------ figures
jn = strrep(P.joint_names, '_', ' ');
f = figure('Visible', 'off', 'Position', [0 0 1500 900]);
tl = tiledlayout(3, 3, 'TileSpacing', 'compact');
for j = 1:3
    nexttile(j); plot(R{1,1}.t, rad2deg(R{1,1}.qr(:,j)), 'k', 'LineWidth', 2); hold on;
    plot(R{1,1}.t, rad2deg(R{1,1}.q(:,j)), '--'); plot(R{1,2}.t, rad2deg(R{1,2}.q(:,j)), ':', 'LineWidth', 1.5);
    title(jn{j}); ylabel('q [deg]'); if j == 1, legend('reference', 'GA-PID', 'LQR'); end
    nexttile(3+j); plot(R{1,1}.t, rad2deg(R{1,1}.e(:,j))); hold on; plot(R{1,2}.t, rad2deg(R{1,2}.e(:,j)));
    ylabel('error [deg]');
    nexttile(6+j); plot(R{1,1}.t, R{1,1}.tau(:,j)); hold on; plot(R{1,2}.t, R{1,2}.tau(:,j));
    ylabel('\tau [Nm]'); xlabel('t [s]');
end
title(tl, 'Simulink (Lagrangian plant): GA-tuned PID vs LQR, nominal rehab task');
exportgraphics(f, fullfile(outDir, 'tracking_nominal.png'), 'Resolution', 150); close(f);

f = figure('Visible', 'off', 'Position', [0 0 900 380]);
v = zeros(numel(sc), 2);
for i = 1:numel(sc), for c = 1:2, v(i, c) = mean(C.(matlab.lang.makeValidName(sc(i).name)).(matlab.lang.makeValidName(ctrl(c).name)).rmse_deg); end, end
bar(categorical({sc.name}, {sc.name}), v); ylabel('mean joint RMSE [deg]'); legend('GA-PID', 'LQR');
title('Robustness comparison (Simulink)');
exportgraphics(f, fullfile(outDir, 'robustness_rmse.png'), 'Resolution', 150); close(f);

f = figure('Visible', 'off', 'Position', [0 0 700 380]);
semilogy(G.history(:,1), G.history(:,2), 'o-', G.history(:,1), G.history(:,3), 's--'); grid on;
xlabel('generation'); ylabel('cost J'); legend('best', 'median'); title('MATLAB ga(): PID gain tuning');
exportgraphics(f, fullfile(outDir, 'ga_convergence.png'), 'Resolution', 150); close(f);
fprintf('Results written to %s\n', outDir);
end
