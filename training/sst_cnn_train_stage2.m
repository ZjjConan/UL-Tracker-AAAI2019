function net = sst_cnn_train_stage2(net, varargin)
    opts.outDir = 'data/snapshot/';
    opts.pairImgDir = 'data/pairs';
    opts.maxNumLoadedFiles = 400;
    
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
    images = vaDir(opts.pairImgDir, 'mat');
    maxNumLoadedFiles = min(numel(images), opts.maxNumLoadedFiles);
    imdb.target = cell(maxNumLoadedFiles, 1);
    imdb.search = cell(maxNumLoadedFiles, 1);
    perms = randperm(numel(images), maxNumLoadedFiles);
    fprintf('[stage 2] sst fine-tuning: find %d discovered pair files\n', maxNumLoadedFiles);
    for i = perms
        fprintf('[state 2] sst fine-tuning: load %d / %d file\n', i, maxNumLoadedFiles);
        load(images{i});
        imdb.target{i} = target;
        imdb.search{i} = search;
    end
    
    imdb.target = cat(4, imdb.target{:});
    imdb.search = cat(4, imdb.search{:});
    imdb.images.set = ones(size(imdb.target, 4), 1);
    
    net = cnn_train_dag(net, imdb, ...
        @(x, y) get_train_batch(x, y, 'gpus', opts.trainOpts.gpus, ...
                    'averageImage', net.meta.normalization.averageImage, ...
                    'augmentFlip', true), ...
                'expDir', opts.outDir, ...
                'val', find(imdb.images.set == 2), ...
                opts.trainOpts);
end



