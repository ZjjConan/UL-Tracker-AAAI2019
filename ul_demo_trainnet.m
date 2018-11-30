% script for self-supervised learning using DCFNet
clc; clear all; close all

%% general settings
opts.imdbDir = 'data/imdb/SR (1994)_imdb.mat';
opts.outDir = 'data/snapshot/DCFNet-SR (1994)/';
opts.saveModelDir = 'data/model/';
opts.gpus = [1];
ul_make_dir(opts.outDir);

imdb = load(opts.imdbDir);

%% setup network
netOpts.lossType = 1;
netOpts.inputSize = 125;
netOpts.padding = 1.5;
net = make_DCFNet_v1(netOpts);

%% train-tracking opts
opts.trackOpts.gpus = opts.gpus;
opts.trackOpts.outputLayer = 'norm1';
opts.trackOpts.numImages = 4; % I in the paper
opts.trackOpts.interval = 10; 
opts.trackOpts.numClips = 400; % V in the paper
opts.trackOpts.numPairs = 16; % K in the paper
opts.trackOpts.minFBAScore = 0.7;
opts.trackOpts.trackBatchSize = 10;
opts.trackOpts.trackingFcn = @track_FBA;
opts.trackOpts.getBatchFcn = @prepare_batch_for_tracking;
opts.trackOpts.gridGenerator = ...
    dagnn.AffineGridGenerator('Ho', netOpts.inputSize, 'Wo', netOpts.inputSize);

opts.trackOpts.grayImage = true;
opts.trackOpts.grayProb = 0.25;
opts.trackOpts.blurImage = true;
opts.trackOpts.blurSigma = 4;
opts.trackOpts.blurProb = 0.25;
opts.trackOpts.rotateImage = true;
opts.trackOpts.rotateProb = 0.25;
opts.trackOpts.rotateRange = [-pi pi]/9;

% train-trainOpts
opts.trainOpts.randpermute = true;
opts.trainOpts.momentum = 0.9;
opts.trainOpts.weightDecay = 0.0005;
opts.trainOpts.learningRate = logspace(-2, -3, 10);
opts.trainOpts.numEpochs = numel(opts.trainOpts.learningRate);
opts.trainOpts.derOutputs = {'objective', 1};
opts.trainOpts.continue = false;

opts.trainOpts.getDataFcn = @build_database;
opts.trainOpts.getBatchFcn = @(x,y) ul_get_train_batch(x, y, 'gpus', [1], ...
    'averageImage', net.meta.normalization.averageImage, ...
    'augFlip', true, 'flipProb', 0.25);

net = ul_cnn_train_dag(net, imdb, opts);

modelPath = @(ep) fullfile(opts.outDir, sprintf('net-epoch-%d.mat', ep));
for i = 1:opts.trainOpts.numEpochs
    load(modelPath(i));
    net = deployDCFNet(dagnn.DagNN.loadobj(net));
    save(fullfile(opts.saveModelDir, ['ul_tracker.e', num2str(i), '.mat']), 'net');
end