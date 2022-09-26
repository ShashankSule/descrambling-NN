%% Compute circular finite differences of X 
function D = finitediff(N, d)
if d == 1
    [~,fy] = gradient(eye(N+2)); 
    D = fy(2:N+1, 2:N+1); 
    D(1,end) = -0.5; 
    D(end,1) = 0.5; 
    D = (1/N)*D; 
else
    Dd = finitediff(N,1); 
    D = Dd^d;
end
end