%% load the bandpass network and weights
net = load('passband_10_50.mat');
band_net = net.band_net; 
W1 = band_net.Layers(2,1).Weights; 
W2 = band_net.Layers(4,1).Weights; 

%% descramble using trace and tikhonov criteria: 
% P1_tr = left_diag(W1, 'max_diag_sum', 1000); 
P2 = left_diag(W2', 'max_diag_sum', 1000);
P2_tr = P2'; 

%% plotting 
% tiledlayout(2,3)
% nexttile
figure();
imagesc(W2); 
title("$$W_2$$", 'interpreter', 'latex', 'FontSize', 20); 
%nexttile
figure();
imagesc(abs(fft2(W2)));
title("2-D DFT of $$ W_2$$", 'interpreter', 'latex', 'FontSize', 20); 
%nexttile
figure();
imagesc(W2*P2_tr); 
title("$$W_2P_2$$", 'interpreter', 'latex', 'FontSize', 20);
%nexttile
figure();
imagesc(abs(fft2(W2*P2_tr))); 
title("2-D DFT of $$W_2P_2$$", 'interpreter', 'latex', 'FontSize', 20);
%cb = colorbar; 
%cb.Layout.Tile = 'east'; 

%% plotting as surface 
x = linspace(0,1,size(W1,2));
y = linspace(0,1,size(W1,1)); 
[X,Y] = ngrid(x,y); 
mesh(x,y,P1_tr*W1)
