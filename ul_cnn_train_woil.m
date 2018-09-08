function net = ul_cnn_train_woil(net, varargin)
    opts.outDir = 'data/snapshot/';
    opts.pairImgDir = 'data/pairs';
    
    opts.trainOpts.learningRate = [];
    opts.trainOpts.numEpochs = numel(opts.trainOpts.learningRate);
    opts.trainOpts.momentum = 0.9;
    opts.trainOpts.weightDecay = 0.0005;
    opts.trainOpts.derOutputs = {'objective', 1};
    opts.trainOpts.gpus = [];
    opts.trainOpts.batchSize = 32;
    opts.trainOpts.continue = true;
    
    [opts, varargin] = vl_argparse(opts, varargin);
        
    % load discovered images
    images = ulDir(opts.pairImgDir, 'mat');
    
    for i = 1:numel(images)
        load(images{i});
        trainData.images.set = ones(size(trainData.target, 4), 1);
        
        trainOpts = opts.trainOpts;
        trainOpts.learningRate = opts.trainOpts.learningRate(i);
        trainOpts.numEpochs = 1;
        trainOpts.continue = false;
        trainOpts.gpus = [1];
        net = cnn_train_dag(net, trainData, ...
                    @(x,y) getTrainBatch(x, y, 'gpus', [1], ...
                            'averageImage', net.meta.normalization.averageImage, ...
                            'augFlip', true, 'flipProb', 0.25), ...
                    'expDir', opts.outDir, ...
                    'val', find(trainData.images.set == 2), ...
                    trainOpts);
    end
end



