%% Fitting a biexponential curve from Wang et al 2013: we need AG for this
time_end = 1.6*500;
t = linspace(-time_end/2,time_end/2,64)'; % Get time data from data generation method
data = v(:,1); % get data
data_fft = fft(data); % Step 1: Get fourier transform
poly_data = (1./abs(data_fft)).^2; % Step 2: Invert power spectral density 
% p = polyfit(sqrt(abs(t(16:48))).*sign(t(16:48)),poly_data(16:48), 2); % Step 3: Fit polynomial 
%% control system toolbox

% zeta = .5;                           % Damping Ratio
% wn = 2;                              % Natural Frequency
% sys = tf(wn^2,[1,2*zeta*wn,wn^2]); 

time_end = 1.6*500;
t = linspace(time_end/64,time_end,64)';
D = 1; 
% y = (-exp(-0.03*t) + exp(-0.0064*t));
v_nd = V(1:64,1); % get data
v_b = V(65:end,1);
y_exp = expcurve(t, 0.18, -0.0055, -0.05, 0.7, -0.3,min(v_nd));
plot(t,y_exp, 'r-')
hold on;
plot(t,v_nd,'bo')

%% compute using nlls

%% first set up variables

% nd variables 
C = optimvar('C',2);
T = optimvar('T', 2);
D = optimvar('D', 1)
v_b = double(v_b - min(v_b)); 
f = C(1)*exp(-T(1)*t) - C(2)*exp(-T(2)*t) + D; 
obj = mean((f-v_b).^2);
lsqproblem = optimproblem("Objective",obj);
x0.C = [0.18*0.7, 0.18*0.3];
x0.T = [0.0055, 0.05]; 

%% solve the problem 
options = optimoptions('fminunc', 'MaxFunctionEvaluations',5000, 'MaxIterations', 5000);
% x0.C = sol.C;
% x0.T = sol.T; 
[sol,fval] = solve(lsqproblem,x0, 'Options',options); 
y = evaluate(f,sol);
%% plot solution 
plot(t,v_b, 'o', 'MarkerSize', 10, 'MarkerFaceColor','blue'); 
hold on; 
plot(t, y, 'r-', 'LineWidth', 3)
set(gca, 'FontSize', 20)
%% use function handle 
function y = expcurve(t,D, T_1, T_2, a_1, a_2, C)
% plot a curve C(a_1e^(T_1)t + a_2e^(T_2)t)
y = D*(a_1*exp(T_1*t) + a_2*exp(T_2*t)) + C;

end