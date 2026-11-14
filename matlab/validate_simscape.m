function V = validate_simscape(Tend)
%VALIDATE_SIMSCAPE Cross-check the SymPy/Lagrangian model against Simscape Multibody.
%   The URDF exported from config/arm_params.json is imported with smimport
%   (independent multibody formulation). Both models swing passively under
%   gravity (zero torque, viscous joint damping) from the same initial pose;
%   joint trajectories are compared. Results -> results/simulink/simscape_validation.*
if nargin < 1, Tend = 1.5; end
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here, 'generated'));
outDir = fullfile(here, '..', 'results', 'simulink'); if ~exist(outDir, 'dir'), mkdir(outDir); end
P = load_arm_params();
q0 = deg2rad([30; 60; 45]);
urdf = fullfile(here, '..', 'isaac', 'assets', 'rehab_arm.urdf');

% ---------------------------------------------------- Simscape Multibody model
mdl = 'rehab_arm_simscape';
if bdIsLoaded(mdl), close_system(mdl, 0); end
smimport(urdf, 'ModelName', mdl);
set_param(mdl, 'StopTime', num2str(Tend), 'Solver', 'ode15s', 'RelTol', '1e-8', 'AbsTol', '1e-10', ...
    'MaxStep', '1e-3', 'ReturnWorkspaceOutputs', 'on');
jb = cell(1, 3);
for j = 1:3
    blk = [mdl '/' P.joint_names{j}];
    jb{j} = blk;
    set_param(blk, 'PositionTargetSpecify', 'on', 'PositionTargetPriority', 'High', ...
        'PositionTargetValue', num2str(rad2deg(q0(j)), 16), 'PositionTargetValueUnits', 'deg', ...
        'VelocityTargetSpecify', 'on', 'VelocityTargetPriority', 'High', 'VelocityTargetValue', '0', ...
        'DampingCoefficient', num2str(P.B(j), 16), 'SpringStiffness', '0', 'SensePosition', 'on');
    try, set_param(blk, 'LowerLimitSpecify', 'off', 'UpperLimitSpecify', 'off'); catch, end %#ok<NOCOM>
    pos = get_param(blk, 'Position'); parent = get_param(blk, 'Parent');
    cv = [parent '/q_conv' num2str(j)];
    add_block('nesl_utility/PS-Simulink Converter', cv, 'Position', pos + [150 0 150 0], 'Unit', 'rad');
    tw = [parent '/q' num2str(j) '_ss'];
    add_block('simulink/Sinks/To Workspace', tw, 'VariableName', sprintf('q%d_ss', j), ...
        'SaveFormat', 'Timeseries', 'Position', pos + [250 0 270 0]);
    ph = get_param(blk, 'PortHandles'); pc = get_param(cv, 'PortHandles');
    sensePort = ph.RConn(end);   % sensing port is added last on the follower side
    add_line(parent, sensePort, pc.LConn(1), 'autorouting', 'on');
    add_line(parent, [get_param(cv, 'Name') '/1'], [get_param(tw, 'Name') '/1'], 'autorouting', 'on');
end
out = sim(mdl);

% ---------------------------------------------------- Lagrangian model (ode45)
f = @(t, x) [x(4:6); arm_M(x(1:3), P.vec) \ (-arm_C(x(1:3), x(4:6), P.vec)*x(4:6) ...
                                              - arm_G(x(1:3), P.vec) - P.B.*x(4:6))];
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
tq = linspace(0, Tend, 1501)';
[~, X] = ode45(f, tq, [q0; 0; 0; 0], opts);

Qs = zeros(numel(tq), 3);
for j = 1:3
    ts = out.get(sprintf('q%d_ss', j));
    Qs(:, j) = interp1(ts.Time, squeeze(ts.Data), tq, 'linear', 'extrap');
end
err = rad2deg(Qs - X(:, 1:3));
V.max_abs_err_deg = max(abs(err), [], 1);
V.rms_err_deg = sqrt(mean(err.^2, 1));
V.q0_deg = rad2deg(q0)'; V.T = Tend;
fid = fopen(fullfile(outDir, 'simscape_validation.json'), 'w');
fprintf(fid, '%s', jsonencode(V, 'PrettyPrint', true)); fclose(fid);

fig = figure('Visible', 'off', 'Position', [0 0 1100 420]);
tiledlayout(1, 2);
nexttile; plot(tq, rad2deg(X(:, 1:3)), 'LineWidth', 2); hold on; set(gca, 'ColorOrderIndex', 1);
plot(tq, rad2deg(Qs), 'k--'); xlabel('t [s]'); ylabel('q [deg]');
title('Passive swing: Lagrangian (solid) vs Simscape Multibody (dashed)');
legend(strrep(P.joint_names, '_', ' '), 'Location', 'best');
nexttile; plot(tq, err); xlabel('t [s]'); ylabel('difference [deg]'); title('Model discrepancy');
exportgraphics(fig, fullfile(outDir, 'simscape_validation.png'), 'Resolution', 150); close(fig);
save_system(mdl, fullfile(here, [mdl '.slx']));
close_system(mdl, 0);
fprintf('Simscape vs Lagrangian: max |err| = [%s] deg\n', num2str(V.max_abs_err_deg, '%.2e '));
end
