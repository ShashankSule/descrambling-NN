%% initialize tapped layers 
L = 6; % set number of layer parameters here 
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
    'Plots','training-progress');
tappednet = trainNetwork(training_x, training_y, tapped_layers, options);
%% descramble! 
% S = tappednet.predict(randn(128,2.5e4));
S = tappednet.predict(data'); 
[P,Q] = descramble(S,1000); 

%% propagate data
w = tappednet.Layers(6).Weights; 
% w = tappednet.predict(eye(128,32));
[L, R] = size(w); 
imagesc(abs(recentered_dft(L)'*P*w*recentered_dft(R)))
% imagesc(P*w)
title("Weight matrix: descrambled on real data", 'FontSize', 20); 
xlabel("Input dimension index", 'FontSize', 20); 
ylabel("Output dimension index", 'FontSize', 20); 
colormap jet 
c = colorbar;
c.FontSize = 20;
%% helpers 
function M = recentered_dft(N) 
    M = diag(exp(1i*pi*(0:N-1)))*dftmtx(N);
end
