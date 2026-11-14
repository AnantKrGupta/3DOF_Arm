function G = tune_pid_ga(popSize, maxGen)
%TUNE_PID_GA Tune the 9 PID gains of rehab_arm_3dof.slx with MATLAB's ga().
%   x = [Kp1 Kp2 Kp3 Ki1 Ki2 Ki3 Kd1 Kd2 Kd3];  fitness = tracking_cost(...)
%   Each fitness evaluation is a full nonlinear Simulink simulation of the
%   rehab task (Fast Restart on). Results -> results/simulink/ga_pid.mat
if nargin < 1, popSize = 40; end
if nargin < 2, maxGen = 30; end
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here, 'generated'));
outDir = fullfile(here, '..', 'results', 'simulink'); if ~exist(outDir, 'dir'), mkdir(outDir); end
mdl = 'rehab_arm_3dof';
if ~exist(fullfile(here, [mdl '.slx']), 'file'), build_rehab_arm_model(mdl); end
load_system(fullfile(here, [mdl '.slx']));

P = load_arm_params();
S = default_sim_vars(P);
lb = [50 50 25    20 20 10     8 8 4];
ub = [800 800 400 800 800 400  50 50 25];

set_param(mdl, 'FastRestart', 'on');
cleanup = onCleanup(@() set_param(mdl, 'FastRestart', 'off'));
history = [];  nEval = 0;  tStart = tic;

    function J = fitness(x)
        S.Kp = diag(x(1:3)); S.Ki = diag(x(4:6)); S.Kd = diag(x(7:9));
        try
            [~, R] = run_sim(mdl, S);
            J = tracking_cost(R, P);
        catch
            J = 1e6;
        end
        nEval = nEval + 1;
    end

    function [state, options, changed] = logger(options, state, flag)
        changed = false;
        if strcmp(flag, 'iter') || strcmp(flag, 'init')
            history(end+1, :) = [state.Generation, min(state.Score), median(state.Score)]; %#ok<AGROW>
            fprintf('  gen %2d  best J = %.4f  median J = %.4f  (%d sims, %.0f s)\n', ...
                state.Generation, min(state.Score), median(state.Score), nEval, toc(tStart));
        end
    end

rng(7);
opts = optimoptions('ga', 'PopulationSize', popSize, 'MaxGenerations', maxGen, ...
    'EliteCount', 2, 'CrossoverFraction', 0.8, 'FunctionTolerance', 1e-6, ...
    'MaxStallGenerations', 12, 'Display', 'off', 'OutputFcn', @logger);
[xbest, Jbest, exitflag] = ga(@fitness, 9, [], [], [], [], lb, ub, [], opts);

G.Kp = diag(xbest(1:3)); G.Ki = diag(xbest(4:6)); G.Kd = diag(xbest(7:9));
G.x = xbest; G.J = Jbest; G.history = history; G.exitflag = exitflag;
G.nEval = nEval; G.wall_time_s = toc(tStart); G.lb = lb; G.ub = ub;
save(fullfile(outDir, 'ga_pid.mat'), '-struct', 'G');
fprintf('GA done: J = %.4f, Kp = [%s], Ki = [%s], Kd = [%s]\n', Jbest, ...
    num2str(xbest(1:3), '%.1f '), num2str(xbest(4:6), '%.1f '), num2str(xbest(7:9), '%.2f '));
end
