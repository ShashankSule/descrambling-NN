%% Generates a weight matrix descrambler for a particular layer in
% a neural network using Tikhonov smoothness criterion. Syntax:
%
%                     P=descramble(S,n_iter,guess)
%
% Parameters:
%
%    S       - a matrix containing, in its columns, the outputs
%              of the preceding layers of the neural network for
%              a (preferably large) number of reasonable inputs
%
%    n_iter  - maximum number of Newton-Raphson interations, 400
%              is generally sufficient
%
%    guess   - [optional] the initial guess for the descrambling
%              transform generator (lower triangle is used), a
%              reasonable choice is a zero matrix (default)
%
% Output:
%
%    P       - descrambling matrix. In the case when the network 
%              is wiretapped before the activation function, i.e.
%
%                             S = Wf(W...f(Wf(WX)))
%
%              matrix P descrambles the output dimension of the
%              left-most W. In the case when the network is wire
%              tapped after the activation function, i.e.
%
%                            S = f(Wf(W...f(Wf(WX))))
%
%              matrix inv(P) descrambles the input dimension of
%              the weight matrix of the subsequent layer.
%   
% i.kuprov@soton.ac.uk
% j.amey@soton.ac.uk
%
% <http://spindynamics.org/wiki/index.php?title=descramble.m>

%% 
function [P,Q]=descramble(S,n_iter,guess)

%Check consistency
grumble(S,n_iter);

% Decide problem dimensions
out_dim=size(S,1);
opt_dim=(out_dim^2-out_dim)/2;

% Lower triangle index array
lt_idx=tril(true(out_dim),-1);

% Default guess is zero
if ~exist('guess','var')
    guess=zeros(opt_dim,1);
else
    guess=guess(lt_idx);
end
   
% Get derivative operator
% [~,D]=fourdif(out_dim,1);
D = finitediff(out_dim,1);

% Precompute some steps, scale, and move to GPU
% SST=gpuArray(S*S'); 
SST = S*S';
SST=out_dim*SST/norm(SST,2);
% DTD=gpuArray(D'*D);
DTD = D'*D; 
DTD=out_dim*DTD/norm(DTD,2);
% U=eye([out_dim out_dim],'gpuArray');
U = eye([out_dim out_dim]);
%% Regularisation signal
function [eta,eta_grad]=reg_sig(q)
    
    % Form the generator
    %Q=zeros([out_dim out_dim],'gpuArray'); 
    Q = zeros([out_dim out_dim]);
    Q(lt_idx)=q; Q=Q-Q';
    
    % Re-use the inverse
    iUpQ=inv(U+Q);
    
    % Run Cayley transform
    P=iUpQ*(U-Q); %#ok<MINV>
    
    % Re-use triple product
    DTDPSST=DTD*P*SST;
    
    % Compute Tikhonov norm
    eta= double(trace(DTDPSST*P'));
    
    % Compute Tikhonov norm gradient
    eta_grad=-2*iUpQ'*DTDPSST*(U+P)';
    
    % Antisymmetrise the gradient
    eta_grad=eta_grad-transpose(eta_grad);
    
    % Extract the lower triangle
    eta_grad=double(eta_grad(lt_idx));
    
    % Move back to CPU
    eta=gather(eta); 
    eta_grad=gather(eta_grad);

end
fun = @reg_sig;
%disp('Pause now')
%pause(3)
% Optimisation
options=optimoptions('fmincon','Algorithm','interior-point','Display','off',...
                     'MaxIterations',n_iter,'MaxFunctionEvaluations',inf,...
                     'FiniteDifferenceType','central','CheckGradients',false,...
                     'SpecifyObjectiveGradient',true,'HessianApproximation',...
                     'finite-difference','SubproblemAlgorithm','cg');
%q=fmincon(@reg_sig,guess,[],[],[],[],-inf(opt_dim,1),+inf(opt_dim,1),[],options);

q=fmincon(fun,guess,[],[],[],[],-inf(opt_dim,1),+inf(opt_dim,1),[],options);

% Form descramble generator
Q=zeros(out_dim); Q(lt_idx)=q; Q=Q-Q';

% Run Cayley transform
P=(U+Q)\(U-Q);

end

% Consistency enforcement
function grumble(S,n_iter)
if (~isnumeric(S))||(~isreal(S))
    error('S must be a real matrix.');
end
if (~isnumeric(n_iter))||(~isreal(n_iter))||...
   (~isscalar(n_iter))||(mod(n_iter,1)~=0)||...
   (n_iter<1)
    error('n_iter must be a positive real integer.');
end
end

function [x, DM] = fourdif(N,m)
%
% The function [x, DM] = fourdif(N,m) computes the m'th derivative Fourier 
% spectral differentiation matrix on grid with N equispaced points in [0,2pi)
% 
%  Input:
%  N:        Size of differentiation matrix.
%  M:        Derivative required (non-negative integer)
%
%  Output:
%  x:        Equispaced points 0, 2pi/N, 4pi/N, ... , (N-1)2pi/N
%  DM:       m'th order differentiation matrix
%
% 
%  Explicit formulas are used to compute the matrices for m=1 and 2. 
%  A discrete Fouier approach is employed for m>2. The program 
%  computes the first column and first row and then uses the 
%  toeplitz command to create the matrix.

%  For m=1 and 2 the code implements a "flipping trick" to
%  improve accuracy suggested by W. Don and A. Solomonoff in 
%  SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
%  The flipping trick is necesary since sin t can be computed to high
%  relative precision when t is small whereas sin (pi-t) cannot.
%
%  S.C. Reddy, J.A.C. Weideman 1998.  Corrected for MATLAB R13 
%  by JACW, April 2003.
 

    x=2*pi*(0:N-1)'/N;                       % gridpoints
    h=2*pi/N;                                % grid spacing
    zi=sqrt(-1);
    kk=(1:N-1)';
    n1=floor((N-1)/2); n2=ceil((N-1)/2);
    if m==0,                                 % compute first column
      col1=[1; zeros(N-1,1)];                % of zeroth derivative
      row1=col1;                             % matrix, which is identity

    elseif m==1,                             % compute first column
      if rem(N,2)==0                         % of 1st derivative matrix
	topc=cot((1:n2)'*h/2);
        col1=[0; 0.5*((-1).^kk).*[topc; -flipud(topc(1:n1))]]; 
      else
	topc=csc((1:n2)'*h/2);
        col1=[0; 0.5*((-1).^kk).*[topc; flipud(topc(1:n1))]];
      end;
      row1=-col1;                            % first row

    elseif m==2,                             % compute first column  
      if rem(N,2)==0                         % of 2nd derivative matrix
	topc=csc((1:n2)'*h/2).^2;
        col1=[-pi^2/3/h^2-1/6; -0.5*((-1).^kk).*[topc; flipud(topc(1:n1))]];
      else
	topc=csc((1:n2)'*h/2).*cot((1:n2)'*h/2);
        col1=[-pi^2/3/h^2+1/12; -0.5*((-1).^kk).*[topc; -flipud(topc(1:n1))]];
      end;
      row1=col1;                             % first row 

    else                                     % employ FFT to compute
      N1=floor((N-1)/2);                     % 1st column of matrix for m>2
      N2 = (-N/2)*rem(m+1,2)*ones(rem(N+1,2));  
      mwave=zi*[(0:N1) N2 (-N1:-1)];
      col1=real(ifft((mwave.^m).*fft([1 zeros(1,N-1)])));
      if rem(m,2)==0,
	row1=col1;                           % first row even derivative
      else
	col1=[0 col1(2:N)]'; 
	row1=-col1;                          % first row odd derivative
      end;
    end;
    DM=toeplitz(col1,row1);
end

function D = finitediff(N, d)
%% Compute circular finite differences of X 
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
% =======================
% Dear ###,
% 
% you may have been wondering, for some of your past publications, 
% why the peer review process was taking such a long time. I would
% like to point out - if I may - that now you know.
%
% Best wishes,
% Ilya.
% =======================
%
% IK's reminder email to junior
% scientists who are dragging
% their feet on a paper review 

