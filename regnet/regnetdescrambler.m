function P = regnetdescrambler(regnet, data, L, pics)
%% initialize tapped layers 
% Input:
% L -- set number of layer parameters here 
% data -- input data
% pics -- save pictures?
% Output: 
% Saves images of descrambled weights 

%% initial preprocessing
% if nargin < 4
%     pics = true;
% end
L = 2;
% L = 4; 
if mod(L,2) ~= 0 
    fprintf('Outermost layer should be affine'); 
    return 
end
tapped_layers = regnet.Layers(1:L);
tapped_layers(L+1) = regressionLayer; 
maxEpochs = 1;
miniBatchSize = 512;
training_x = randn(tapped_layers(1).InputSize,1); 
% training_y = randn(tapped_layers(L-1).OutputSize,1); 
training_y = randn(tapped_layers(L).OutputSize,1);
%% initialize network 
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-10, ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','none');
tappednet = trainNetwork(training_x, training_y, tapped_layers, options);
%% descramble! 
% S = tappednet.predict(randn(128,2.5e4));
S = tappednet.predict(ndbdata'); 
[P,~] = descramble(S,10000);  

%% propagate data
L = 2;
w = tappednet.Layers(L).Weights; 
w = w(:,65:128); % pick the nd half
% w = tappednet.predict(eye(128,32));W\W
[l, r] = size(w); 
%imagesc(abs(recentered_dft(l)'*P*w*recentered_dft(r)))
%imagesc(P*w)
% w = J; 
if size(w,2) > size(w,1)
    xx = 1; 
    yy = 2;
else
   xx = 2; 
   yy = 1; 
end
pics = false;
% pics = true; 
figure(1); 
set(gca, 'FontSize', 20)
tiledlayout(yy,xx)
nexttile
imagesc(w);
title("Weights: Raw", 'FontSize', 20); 
xlabel("Input index", 'FontSize', 20); 
ylabel("Output index", 'FontSize', 20); 
nexttile;
imagesc(P*w); 
title("Weights: Descrambled", 'FontSize', 20); 
xlabel("Input index", 'FontSize', 20); 
ylabel("Output index", 'FontSize', 20); 
colormap parula 
c = colorbar;
c.FontSize = 20;
c.Layout.Tile = 'east';
filestring = strcat(pwd(), '/jun10/Weights_', num2str(L), '.png'); 
if pics
    saveas(gcf, filestring);
end
% clf;

figure(2)
set(gca, 'FontSize', 20)
tiledlayout(yy,xx); 
nexttile
imagesc(abs(recentered_dft(l).'*w*recentered_dft(r))); 
title("FFT Weights: Raw", 'FontSize', 20); 
xlabel("Input index", 'FontSize', 20); 
ylabel("Output index", 'FontSize', 20); 
nexttile
imagesc(abs(recentered_dft(l).'*P*w*recentered_dft(r))); 
title(["FFT Weights:";"Descrambled"], 'FontSize', 20); 
xlabel("Input index", 'FontSize', 20); 
ylabel("Output index", 'FontSize', 20); 
colormap parula 
c = colorbar;
c.FontSize = 20;
c.Layout.Tile = 'east'; 
if pics
    saveas(gcf, strcat(pwd(), '/jun10/FFTs_', num2str(L), '.png'));
end
% clf reset; 

desc = P*w; 
[u_r, sigma_r, v_r] = svd(w); 
[u, sigma, v] = svd(desc); 

figure(3);
set(gca, 'FontSize', 20)
tiledlayout(yy,xx);
nexttile
imagesc(u_r); 
title("Left SV: Raw", 'FontSize', 20); 
xlabel("Output index", 'FontSize', 20); 
ylabel("Output index", 'FontSize', 20); 
nexttile
imagesc(u); 
title(["Left SV"; "Descrambled"], 'FontSize', 20); 
xlabel("Output index", 'FontSize', 20); 
ylabel("Output index", 'FontSize', 20); 
colormap parula
c = colorbar;
c.FontSize = 20;
c.Layout.Tile = 'east';
if pics
    saveas(gcf, strcat(pwd(), '/jun10/LeftSV_', num2str(L), '.png'));
end
% clf reset; 

figure(4)
set(gca, 'FontSize', 20)
plot(diag(sigma_r), 'r-', 'LineWidth',2); 
hold on;
plot(diag(sigma), 'b-', 'LineWidth', 2);
legend('Raw', 'Descrambled');
title('Comparing decay of singular values', 'FontSize', 20); 
if pics
    saveas(gcf, strcat(pwd(), '/jun10/Decay_', num2str(L), '.png'));
end
% clf reset; 

figure(5)
tiledlayout(yy,xx)
nexttile
plot(u_r(:,1), 'r-', 'LineWidth',2);
hold on;
plot(u_r(:,2), 'b-', 'LineWidth',2); 
legend('First left singular vector', 'Second left singular vector'); 
title("Singular vectors: Raw", 'FontSize', 20);
nexttile
plot(u(:,1), 'r-', 'LineWidth',2);
hold on;
plot(u(:,2), 'b-', 'LineWidth',2); 
legend('First left singular vector', 'Second left singular vector'); 
title("Singular vectors: Descrambled", 'FontSize', 20); 
if pics
    saveas(gcf, strcat(pwd(), '/jun10/SingLeft_', num2str(L), '.png'));
end
% clf reset; 

figure(6)
tiledlayout(yy,xx)
nexttile
plot(abs(fft(u_r(:,1))), 'r-', 'LineWidth',2);
hold on;
plot(abs(fft(u_r(:,2))), 'b-', 'LineWidth',2); 
legend('First left singular vector FFT', 'Second left singular vector'); 
title("Singular vector FFT: Raw", 'FontSize', 20);
nexttile
plot(abs(fft(u(:,1))), 'r-', 'LineWidth',2);
hold on;
plot(abs(fft(u(:,2))), 'b-', 'LineWidth',2); 
legend('First left singular vector', 'Second left singular vector'); 
title(["Singular vector FFT:"; "Descrambled"], 'FontSize', 20); 
if pics
    saveas(gcf, strcat(pwd(), '/jun10/SingLeftFFT_', num2str(L), '.png'));
end
% clf reset; 


%% helpers 
function M = recentered_dft(N) 
    M = diag(exp(1i*pi*(N-1)*(1/N)*(0:N-1)))*dftmtx(N);
end

end