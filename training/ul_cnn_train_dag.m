function net = ul_cnn_train_dag(net, imdb, varargin)
    opts.outDir = 'data/snapshot/';
    opts.saveDir = 'models/';
    opts.arch = 'DCFNet';
    opts.gpus = [];
    
    % train options
    opts.trainOpts.randpermute = false;
    opts.trainOpts.momentum = 0.9;
    opts.trainOpts.weightDecay = 0.0005;
    opts.trainOpts.maxIters = 10000;
    opts.trainOpts.learningRate = logspace(-2, -5, 50);
    opts.trainOpts.numEpochs = numel(opts.trainOpts.learningRate);
    opts.trainOpts.derOutputs = {'objective', 1};
    opts.trainOpts.solver = [];
    opts.trainOpts.nesterovUpdate = false;
    opts.trainOpts.continue = false;
    opts.trainOpts.batchSize = 32;
    opts.trainOpts.getBatchFcn = @ul_get_train_batch;
    opts.trainOpts.getDataFcn = @build_database;
    
    % track options
    opts.trackOpts.gpus = [];
    opts.trackOpts.outputLayer = 'conv1s';
    opts.trackOpts.numClips = 400;
    opts.trackOpts.numImages = 25;
    opts.trackOpts.interval = 10;
    opts.trackOpts.trackBatchSize = 16;
    opts.trackOpts.numPairs = 32;
    opts.trackOpts.minFBAScore = 0.7;
    opts.trackOpts.trackingFcn = @track_FBA;
    opts.trackOpts.getBatchFcn = @prepare_batch_for_tracking;
    opts.trackOpts.grayImage = true;
    opts.trackOpts.grayProb = 0.25;
    opts.trackOpts.blurImage = true;
    opts.trackOpts.blurSigma = 2;
    opts.trackOpts.blurProb = 0.25;
    opts.trackOpts.rotateImage = false;
    opts.trackOpts.rotateProb = 0.25;
    opts.trackOpts.rotateRange = [0 0];
    opts.trackOpts.gridGenerator = [];
    
    [opts, varargin] = vl_argparse(opts, varargin);
    
    ul_make_dir(opts.outDir);
    
    modelPath = @(ep) fullfile(opts.outDir, sprintf('net-epoch-%d.mat', ep));
    
    if opts.trainOpts.continue 
        startEpoch = findLastCheckpoint(opts.outDir);
        fprintf('%s: resuming by loading epoch %d\n', mfilename, startEpoch) ;
        [net, state, stats] = loadState(modelPath(startEpoch)) ;
    else
        startEpoch = 0;
        state = [] ; 
        stats = [];
    end
    
    learningRate = opts.trainOpts.learningRate;
    for e = startEpoch+1:opts.trainOpts.numEpochs        
        opts.trainOpts.learningRate = learningRate(e);
        
        if isempty(state) || isempty(state.solverState)
            state.solverState = cell(1, numel(net.params)) ;
            state.solverState(:) = {0} ;
        end
        
        prepareGPUs(opts, e == startEpoch + 1) ;
        
        if opts.trainOpts.randpermute
            perm = randperm(numel(imdb.images.data));
        else
            perm = 1:numel(imdb.images.data);
        end
        
        if opts.gpus >= 1
            net.move('gpu');
            for i = 1:numel(state.solverState)
                s = state.solverState{i} ;
                if isnumeric(s)
                    state.solverState{i} = gpuArray(s) ;
                elseif isstruct(s)
                    state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
                end
            end
        end
        
        numTracks = ceil(numel(perm) / opts.trackOpts.numClips);
        stats.num = 0 ; % return something even if subset = []
        stats.time = 0 ;
        start = tic;
        for t = 1:numTracks
            bstart = (t-1) * opts.trackOpts.numClips + 1;
            bend = min(t * opts.trackOpts.numClips, numel(imdb.images.data));
            trainData = opts.trainOpts.getDataFcn(imdb, net, perm(bstart:bend), opts.trackOpts, e); 
            
            numData = size(trainData.target, 4);
            trainBatches = ceil(numData / opts.trainOpts.batchSize);
            net.mode = 'normal' ;
            for it = 1:trainBatches
                bstart = (it - 1) * opts.trainOpts.batchSize + 1;
                bend = min(it * opts.trainOpts.batchSize, numData);
                batch = bstart:bend;
                inputs = opts.trainOpts.getBatchFcn(trainData, batch);
                net.eval(inputs, opts.trainOpts.derOutputs) ;
                state = accumulateGradients(net, state, opts.trainOpts, numel(batch), []) ;
                
                % Get statistics.
                time = toc(start) ;
                batchTime = time - stats.time ;
                stats.num = stats.num + numel(batch);
                stats.time = time ;
                stats = extractStats(stats, net) ;
                currentSpeed = numel(batch) / batchTime ;
                averageSpeed = stats.num / time ;

                fprintf('UL-tracker: training: epoch %02d: %3d/%3d:', e, it, trainBatches) ;
            
                fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
                for f = setdiff(fieldnames(stats)', {'num', 'time', 'train'})
                    f = char(f) ;
                    fprintf(' %s: %.3f', f, stats.(f)) ;
                end
                fprintf('\n') ;
            end
            % save the memory
            clear trainData;
        end
        saveShot(net, state, stats, modelPath(e));
    end
end

