%% Get the net (we assume you already have a bunch of net inputs) 
net = load('/home/ssule25/Documents/spinach_2_6_5625/My_experiments/2_Layer_DeerNet/sigmoid/2_layer_DEERNET_sigmoid.mat');
net_s = net.net; 
%% Make a new net that is just the first layer of the network
layers_1 = net.Layers(1:2,1); 
layers_1(3,1) = net.Layers(end,1); 
net_1 = assembleNetwork(layers_1); 
layers_2 = net.Layers(1:3,1); 
layers_2(4,1) = net.Layers(end,1); 
net_2 = assembleNetwork(layers_2);

%% Now prepare for descramble! 
lib = load('/home/ssule25/Documents/spinach_2_6_5625/My_experiments/2_Layer_DeerNet/example_set_2layer.mat'); 
inputs = lib.library.deer_noisy_lib; 
inputs = inputs'; 
inputs = gpuArray(inputs); 
descr_inputs = net_2.predict(inputs); 

%% Now descramble! 
[P_2,Q_2] = descramble(descr_inputs', 1000);
%save('descram.mat', 'P'); 
%save('anti_sym.mat', 'Q'); 

%% Recenter the fourier transform 

D_80 = diag(exp(1i*pi*(1:80)))*dftmtx(80); 
D_256 = diag(exp(1i*pi*(1:256)))*dftmtx(256); 
W_1 = net_1.Layers(2,1).Weights; 
W_2 = net.Layers(4,1).Weights; 
descram_W_1 = P_1*W_1; 
descram_W_2 = W_2*P_2'; 
%% now plot! 
figure(); 
tiledlayout(2,1); 
nexttile
imagesc(W_2');
title("Raw Weight matrix"); 
axis xy
nexttile
imagesc(descram_W_2'); 
title("Descrambled Weight Matrix"); 
axis xy 
% nexttile
% imagesc(abs(D_80'*W_1*D_256)');
% title("$F^+ W F^-$", 'Interpreter', 'latex');  
% axis xy
% nexttile
% imagesc(abs(D_80'*descram_W_1*D_256)');
% title("$F^+ PW F^-$", 'Interpreter', 'latex');  
% axis xy
%saveas(gcf, 'Layer1.png');

%% second alyer 

figure(); 
[U, Sigma, V] = svd(descram_W_2'*descram_W_2); 
tiledlayout(2,2); 
nexttile
imagesc(W_2');
title("Raw Weight matrix"); 
axis xy
nexttile
imagesc(descram_W_2'); 
title("Descrambled Weight Matrix"); 
axis xy 