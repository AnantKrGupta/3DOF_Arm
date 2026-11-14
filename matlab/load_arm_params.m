function P = load_arm_params(cfgFile)
%LOAD_ARM_PARAMS Read config/arm_params.json (shared with Python & Isaac Sim).
%   P.vec is the dynamic parameter vector [L1 L2 c1 c2 m1 m2 mp r1 r2 g]
%   expected by the SymPy-generated arm_M / arm_C / arm_G functions.
if nargin < 1
    here = fileparts(mfilename('fullpath'));
    cfgFile = fullfile(here, '..', 'config', 'arm_params.json');
end
d = jsondecode(fileread(cfgFile));
P.L1 = d.links.L1_upper_arm_m;      P.L2 = d.links.L2_forearm_hand_m;
P.r1 = d.links.r1_upper_arm_radius_m; P.r2 = d.links.r2_forearm_radius_m;
P.m1 = d.mass.m1_upper_arm_kg;      P.m2 = d.mass.m2_forearm_kg;
P.mp = d.mass.mp_payload_kg;
P.c1 = d.com_ratio.c1_upper_arm;    P.c2 = d.com_ratio.c2_forearm;
P.g  = d.gravity_m_s2;
P.limits  = deg2rad(d.joint_limits_deg);        % 3x2
P.tau_max = d.tau_max_Nm(:);                    % 3x1
P.qdot_max = deg2rad(d.qdot_max_deg_s(:));
P.B       = d.viscous_Nms_rad(:);
P.joint_names = d.joint_names;
P.vec = [P.L1 P.L2 P.c1 P.c2 P.m1 P.m2 P.mp P.r1 P.r2 P.g]';
end
