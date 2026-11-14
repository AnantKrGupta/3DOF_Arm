%RUN_ALL One-click MATLAB/Simulink part of the study.
%   Prerequisite: python python/scripts/02_derive_dynamics.py  (generates matlab/generated/*.m)
%   1) build Simulink model  2) LQR design  3) GA PID tuning  4) comparison
%   5) (optional) Simscape Multibody cross-check of the Lagrangian model
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here, 'generated'));
build_rehab_arm_model();
design_lqr();
tune_pid_ga(40, 30);
compare_controllers();
try
    validate_simscape();
catch err
    warning('Simscape validation skipped: %s', err.message);
end
