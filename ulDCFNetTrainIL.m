% script for self-supervised learning using dcfnet
clc;

videoName = 'SR (1994)';
etime = 1;

%% general settings
opts.imdbDir = ['data/imdb/' videoName '_imdb.mat'];
opts.saveModelName = ['DCFNet - ' videoName ' - nil'];

opts.outDir = 'data/snapshot/';
opts.saveModelDir = 'F:\Research\tracker_zoo\DCFNet\model';
opts.outPairImgDir = 'data/pairs/';
opts.saveInternalPairs = false;
opts.gpus = [1];

opts.outDir = fullfile(opts.outDir, [opts.saveModelName ' - r' num2str(etime)]);
ulMakeDir(opts.outDir);

%% setup network
netOpts.lossType = 1;
netOpts.inputSize = 125;
netOpts.padding = 2;
netOpts.averageImage = reshape(single([123,117,104]), [1,1,3]);
net = initDCFNet(netOpts);
net.meta.normalization.averageImage = netOpts.averageImage;

%% online tracking opts
opts.trackOpts.gpus = opts.gpus;
opts.trackOpts.visualization = 0;
opts.trackOpts.trackingFeatrLayer = 'conv1s';
opts.trackOpts.numImagesPerClip = 4;
opts.trackOpts.maxInterval = 10;
opts.trackOpts.trackingNumClips = 400;
opts.trackOpts.selectNums = 16;
opts.trackOpts.selectThre = 0.7;
opts.trackOpts.FBABatchSize = 10;
opts.trackOpts.trackingFcn = @DCFNetFBWTracking;
opts.trackOpts.getBatchFcn = @getBatchFromWhole;
opts.trackOpts.FBA = true;
opts.trackOpts.gridGenerator = ...
    dagnn.AffineGridGenerator('Ho', netOpts.inputSize, ...
    'Wo', netOpts.inputSize);

opts.trackOpts.grayImage = true;
opts.trackOpts.grayProb = 0.25;
opts.trackOpts.blurImage = true;
opts.trackOpts.blurSigma = 4;
opts.trackOpts.blurProb = 0.25;
opts.trackOpts.rotateImage = true;
opts.trackOpts.rotateProb = 0.25;
opts.trackOpts.rotateRange = [-pi pi]/3;

% trainOpts
opts.trainOpts.randpermute = true;
opts.trainOpts.momentum = 0.9;
opts.trainOpts.weightDecay = 0.0005;
opts.trainOpts.learningRate = logspace(-2, -3, 10);
opts.trainOpts.numEpochs = numel(opts.trainOpts.learningRate);
opts.trainOpts.derOutputs = {'objective', 1};
opts.trainOpts.continue = true;

net = ul_cnn_train_woil(net, opts);

modelPath = @(ep) fullfile(opts.outDir, sprintf('net-epoch-%d.mat', ep));
for i = 1:opts.trainOpts.numEpochs
    load(modelPath(i));
    net = deployDCFNet(dagnn.DagNN.loadobj(net));
    save(fullfile(opts.saveModelDir, [opts.saveModelName ' - e' num2str(i) '.mat']), 'net');
end