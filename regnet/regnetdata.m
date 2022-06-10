%% generate data for the SNR 900 lambda neg 4 regnet
net = load(strcat(pwd(), '/matdata/Week12_SNR10000_lambdaneg4_NDNDvNDB_NN_Week12_NDBPE_minMetric2.mat'));
data = load(strcat(pwd(), '/_results/_dataSets/_snr900/Week12_SNR900_lambdaneg4_NDNDvNDB_ValidationData.mat'));
data = data.data(:,1:128); 
regnet = regnetconverter(net,128); 
%% done! now run the ton of descramblers and vomit some data out!
descramblers = {}; 
for i=1:4
    L = 2*i;
    descramblers{i} = regnetdescrambler(regnet, data, L, true);
end

