function regnet = regnetconverter(net, inputdim)
%% create nn for initialization 
% net = load(...) 
training_x = randn(inputdim,1);
training_y = randn(2,1);
layernames = fieldnames(net); 
layers = [...
          sequenceInputLayer(size(training_x,1), 'Name', 'input')
          fullyConnectedLayer(32, 'Name', 'inputlayer', ...
                              'WeightLearnRateFactor', 0.0,...
                              'BiasLearnRateFactor', 0.0)
          reluLayer('Name', 'relu_input') 
          fullyConnectedLayer(256, 'Name', 'hidden1', ...
                              'WeightLearnRateFactor', 0.0,...
                              'BiasLearnRateFactor', 0.0) 
          reluLayer('Name', 'relu_hidden1')
          fullyConnectedLayer(256, 'Name', 'hidden2', ...
                              'WeightLearnRateFactor', 0.0,...
                              'BiasLearnRateFactor', 0.0)
          reluLayer('Name', 'relu_hidden2') 
          fullyConnectedLayer(32, 'Name', 'hidden3', ...
                              'WeightLearnRateFactor', 0.0,...
                              'BiasLearnRateFactor', 0.0) 
          reluLayer('Name', 'relu_hidden3') 
          fullyConnectedLayer(2, 'Name', 'output', ...
                              'WeightLearnRateFactor', 0.0,...
                              'BiasLearnRateFactor', 0.0)
          reluLayer('Name', 'relu_output') 
          regressionLayer];
%% set weights from pretrained net
counter = 1; 
for i=1:length(layers)
    if isprop(layers(i), 'Weights')
        layers(i).Weights = net.(layernames{counter}); 
        counter = counter + 1; 
    end
    if isprop(layers(i), 'Bias')
        layers(i).Bias = net.(layernames{counter})';
        counter = counter + 1; 
    end
end
%%
maxEpochs = 1;
miniBatchSize = 512;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','none');
regnet = trainNetwork(training_x, training_y, layers, options); 

% %% propagate data
% w = regnet.Layers(2).Weights; 
% [L, R] = size(w); 
% imagesc(abs(recentered_dft(L)'*w*recentered_dft(R)))
%% helpers 
function M = recentered_dft(N) 
    M = diag(exp(1i*pi*(0:N-1)))*dftmtx(N);
end
end
