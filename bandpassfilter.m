%% Bandpass-filtering network 
% here's where we create a data set for a bandpass-filtering ff nn 

%% create ground signal, test and training data
fs = 1024;
t = 0:1/fs:1;
ground_signal = [0.4 0.2 0.4]*sin(2*pi*[50 150 250]'.*t);
ground_signal = ground_signal';
training_x = repmat(ground_signal, 1, 1000) + randn(size(ground_signal,1),1000)*1.1;
training_y = bandpass(training_x, [100 200], fs); 
ground_signal_testing = [0.4 0.2 0.4]*sin(2*pi*[92 170 210]'.*t);
ground_signal_testing = ground_signal_testing' ; 
testing_x = repmat(ground_signal_testing, 1, 1000) + randn(size(ground_signal_testing,1),1000)*1.1;
testing_y = bandpass(testing_x, [100 200], fs); 


%% Create network here

% Use deep network designer to create layer object layer_1 
layers = [...
          sequenceInputLayer(size(training_x,1)) 
          fullyConnectedLayer(size(training_x,1))
          reluLayer
          fullyConnectedLayer(size(training_x,1))
          tanhLayer
          regressionLayer];
      
maxEpochs = 100;
miniBatchSize = 64;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

band_net = trainNetwork(training_x, training_y, layers, options); 

%% Descramble here

bp_layers = band_net.Layers(1:2, 1); 
bp_layers(3,1) = regressionLayer; 
bp_layer_1 = assembleNetwork(bp_layers); 
S = bp_layer_1.predict(training_x); 
guess_gen = tril(randn(size(S,1), size(S,1)));
guess = guess_gen - guess_gen'; 
[P,Q] = descramble(S, 1000, guess)
