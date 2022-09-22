%% generate data for the SNR 900 lambda neg 4 regnet
net = load(strcat(pwd(), '/matdata/Week12_SNR10000_lambdaneg4_NDNDvNDB_NN_Week12_NDBPE_minMetric2.mat'));
data = load(strcat(pwd(), '/_results/_dataSets/_snr10000/Week12_SNR10000_lambdaneg4_NDNDvNDB_TrainingData.mat'));
ndbdata = data.data(:,1:128); 
ndnddata = [ndbdata(:, 1:64) ndbdata(:, 1:64)];
regnet = regnetconverter(net,128); 

%% done! now run the ton of descramblers and vomit some data out!
descramblers = {}; 
for i=1:4
    L = 2*i;
    descramblers{i} = regnetdescrambler(regnet, data, L, true);
end

%% analyze two copies of trained nets 
% ndb = load(strcat(pwd(), '/matdata/Week12_SNR10000_lambdaneg4_NDNDvNDB_NN_Week12_NDBPE_minMetric2.mat'));
% ndnd = load(strcat(pwd(), '/matdata/Week9_SNR10000_lambdaneg1_NDNDvNDB_NN_Week9_NDND_NoDiscardPE_minMetric2.mat'));
net2 = load(strcat(pwd(), '/matdata/Week9_SNR10000_lambdaneg1_NDNDvNDB_NN_Week8_NDND_NoDiscardPE_minMetric (2)2.mat'));
%% compute singular values of each weight matrix
layernames = fieldnames(ndb); 
ndbsvd = struct();
ndndsvd = struct(); 
%% compute svd's
for i = 1:5
    index = 2*i - 1;
    [undb,sigmandb,vndb] = svd(ndb.(layernames{index})); 
    [undnd, sigmandnd, vndnd] = svd(ndnd.(layernames{index})); 
    ndbsvd(i).U = undb; 
    ndbsvd(i).Sigma = sigmandb; 
    ndbsvd(i).V = vndb;
    ndndsvd(i).U = undnd; 
    ndndsvd(i).Sigma = sigmandnd; 
    ndndsvd(i).V = vndnd; 
end

%% plot them svd's! 
for i=1:5
    figure(i)
    tiledlayout(1,3); 
    nexttile 
    plot(diag(ndbsvd(i).Sigma), 'bo-'); 
    hold on; 
    plot(diag(ndndsvd(i).Sigma), 'ro-');
    xlabel("Index", 'FontSize', 20); 
    ylabel("Singular value", 'FontSize', 20); 
    layername = strcat(num2str(i), "th layer"); 
    title(strcat("Singular values: ", layername), 'FontSize', 20); 
    nexttile
    plot(ndbsvd(i).V(:,1:2), 'o-'); 
    title("(ND,Reg)", 'FontSize', 20); 
    legend('First singular vector', 'Second singular vector');
    nexttile
    plot(ndndsvd(i).V(:,1:2), 'o-'); 
    title("(ND,ND)", 'FontSize',20); 
    legend('First singular vector', 'Second singular vector'); 
    set(gcf, 'units', 'inches', 'position', [0.5,0.5,12,10]);
    saveas(gcf, strcat(num2str(i), '_1.png'))
end
