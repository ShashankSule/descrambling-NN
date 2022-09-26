%% design least squares problems to find exponential interpolants

%% set up network 
net = load(strcat(pwd(), '/matdata/Week12_SNR900_lambdaneg1_NDNDvNDB_NN_Week12_NDNDPE_minMetric (1)2.mat'));
W = net.inputMapweight; 
[U, Sigma, V]=svd(W); 

%% get data and dimensions
SNR = 900; 
n = 3;
data = V(:,n); 
N = max(size(data));
k = 64; % point at which we split the data
v_nd = data(1:k); % noisy part 
v_b = data(k+1:end); % smooth part 
time_end = 1.6*500;
t = linspace(time_end/64,time_end,64)'; % get time steps 

%% solve for the interpolants

[i_nd, sol_nd] = find_exp(v_nd, t); % get nd interpolant 
[i_b, sol_b] = find_exp(-v_b, t); % get b interpolant 

%% plot the interpolants together 
plot(data, 'o', 'MarkerSize', 10, 'MarkerFaceColor','blue'); 
hold on; 
plot([i_nd; -i_b], 'r-', 'LineWidth', 3)
title(strcat('SNR: ', num2str(SNR), ', $n = ', num2str(n), '$'), 'FontSize', 20, 'Interpreter','latex')
set(gca, 'FontSize', 20);
leg = legend({'Right Singular Vector', 'Bi-exponential Fit'}); 
set(leg, 'Interpreter', 'latex')

%% helpers 
function [y, sol] = find_exp(x,t)

%find a continuous interpolant C_1*exp(-T_1x) + C_2*exp(-T_2x) + D to x 

%% First convert to double 
x = double(x); 
t = double(t); 

%% set up options + vars 
C = optimvar('C', 2);
T = optimvar('T', 2);
D = optimvar('D', 1);
f = C(1)*exp(-T(1)*t) - C(2)*exp(-T(2)*t) + D; 
tr = min(x); 
x = x - tr; 
obj = mean((f-x).^2);
lsqproblem = optimproblem("Objective",obj);
x0.C = [0.18*0.7, 0.18*0.3];
x0.T = [0.0055, 0.05]; 
x0.D = -min(x);

%% now solve 
options = optimoptions('fminunc', 'MaxFunctionEvaluations',5000, 'MaxIterations', 5000);
% x0.C = sol.C;
% x0.T = sol.T; 
[sol,~] = solve(lsqproblem,x0, 'Options',options); 
y = evaluate(f,sol) + tr;

end

%% exponential curve function 
function y = expcurve(t,D, T_1, T_2, a_1, a_2, C)
% plot a curve C(a_1e^(T_1)t + a_2e^(T_2)t)
y = D*(a_1*exp(T_1*t) + a_2*exp(T_2*t)) + C;

end