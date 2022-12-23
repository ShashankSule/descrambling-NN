%% Filtering a brownian motion 

% drift = @(t) [0.4 0.2 0.4]*sin(2*pi*t*[50 150 250]'); % underlying signal
% diffusion = @(t) 1; %diffusion temperature
% BM = bm(drift, diffusion); % generate bunch of brownian motions 
% fs = 1024; % signal sampling rate
% training_x = BM.simulate(fs-1, 'nTrials',1e6); % generate BM 
% training_x = training_x(:,1,:);
% training_y = bandpass(training_x, [140 160], fs); 

%% Using random matrix ensemble as data
fs = 1024; % signal sampling rate
training_x = normrnd(0,1,[fs,1e4]);
training_y = bandpass(training_x, [10 50], fs); 
% training_x = gpuArray(training_x); 
% training_y = gpuArray(training_y); 
%% Create network here

% Use deep network designer to create layer object layer_1 
layers = [...
          sequenceInputLayer(size(training_x,1)) 
          fullyConnectedLayer(size(training_x,1), 'BiasInitializer', 'zeros', 'BiasLearnRateFactor', 0.0)
          reluLayer
          fullyConnectedLayer(size(training_x,1), 'BiasInitializer', 'zeros', 'BiasLearnRateFactor', 0.0)
          tanhLayer
          regressionLayer];
      
maxEpochs = 100;
miniBatchSize = 512;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

band_net = trainNetwork(training_x, training_y, layers, options); 
