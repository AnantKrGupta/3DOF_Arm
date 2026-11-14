function [qr, qrd, qrdd] = rehab_task(t)
%REHAB_TASK Minimum-jerk rehabilitation task (mirrors python/rehab_arm/trajectory.py).
%   Waypoints [abduction flexion elbow] (deg): rest -> forward reach ->
%   hand-to-mouth -> lateral raise -> rest. 2 s moves, 0.5 s holds.
%#codegen
Wdeg = [10 10 20; 15 60 30; 20 55 125; 75 20 40; 10 10 20];
move = [0 2 2 2 2];
hold = [0.5 0.5 0.5 0.5 0.5];
W = Wdeg * pi/180;
qr = W(1,:)'; qrd = zeros(3,1); qrdd = zeros(3,1);
ts = 0;
for i = 1:size(W,1)
    if i > 1 && t >= ts
        T = move(i);
        tau = min(max((t - ts)/T, 0), 1);
        s   = 10*tau^3 - 15*tau^4 + 6*tau^5;
        sd  = (30*tau^2 - 60*tau^3 + 30*tau^4)/T;
        sdd = (60*tau - 180*tau^2 + 120*tau^3)/T^2;
        dq = (W(i,:) - W(i-1,:))';
        qr = W(i-1,:)' + s*dq;  qrd = sd*dq;  qrdd = sdd*dq;
    end
    ts = ts + move(i) + hold(i);
end
end
