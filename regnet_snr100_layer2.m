%% descrambling regnet
W = load('weight_2_wk12_snr10000.mat'); 
W = W.w2; 

%% descramble now!!
P_left_trsq = left_diag(W, 'max_diag_normsq', 1000);
P_right_trsq = left_diag(W', 'max_diag_normsq', 1000);
P_right_trsq = P_right_trsq'; 

%% plot reg net! 
figure(); 
imagesc(W); 
title("$$W_2$$", 'interpreter', 'latex', 'FontSize', 20);
figure(); 
imagesc(W*P_right_tr); 
title("$$W_2 P_2$$", 'interpreter', 'latex', 'FontSize', 20);
figure(); 
imagesc(P_left_tr*W); 
title("$$Q_2 W_2$$", 'interpreter', 'latex', 'FontSize', 20);
figure(); 
plot((1/apw)*diag(P_left_tr*W), 'ro-'); 
hold on; 
plot((1/aw)*diag(W'*W), 'bo-');
l = legend('diag(QW)', 'diag($$W^{\top}W)$$', 'FontSize', 15);
set(l, 'Interpreter', 'latex')
