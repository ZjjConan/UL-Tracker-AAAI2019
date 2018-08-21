function net = ul_cnn_train_dag(net, imdb, varargin)
    opts.outDir = 'data/snapshot/';
    opts.outPairImgDir = 'data/pairs';
    opts.saveModelDir = 'models/';
    opts.arch = 'DCFNet';
    opts.saveInternalPairs = false;
    opts.gpus = [];
    
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
    opts.trainOpts.getBatchFcn = @getTrainBatch;
    opts.trainOpts.getDataFcn = @dcfnet_get_data_online;
    
    opts.trackOpts.visualization = false;
    opts.trackOpts.gpus = [];
    opts.trackOpts.trackingFeatrLayer = 'conv1s';
    opts.trackOpts.trackingNumPerEpoch = 20;
    opts.trackOpts.numImagesPerClip = 25;
    opts.trackOpts.maxInterval = 10;
    opts.trackOpts.FBWBatchSize = 16;
    opts.trackOpts.selectNums = 32;
    opts.trackOpts.selectThre = 0.7;
    opts.trackOpts.trackingFcn = @DCFNetFBWTracking;
    opts.trackOpts.getBatchFcn = @getTrackingBatchFromClip;
    [opts, varargin] = vl_argparse(opts, varargin);
    
    ulMakeDir(opts.outDir);
    
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
        
        numClips = ceil(numel(imdb.images.data) / opts.trackOpts.trackingNumPerEpoch);
        stats.num = 0 ; % return something even if subset = []
        stats.time = 0 ;
        start = tic;
        for t = 1:opts.trackOpts.trackingNumPerEpoch
            bstart = (t-1) * numClips + 1;
            bend = min(t * numClips, numel(imdb.images.data));
            trainData = opts.trainOpts.getDataFcn(imdb, net, perm(bstart:bend), opts.trackOpts, e); 
            
            numData = size(trainData.search, 4);
            if opts.trainOpts.randpermute
               order = randperm(numData);
               trainData.search = trainData.search(:,:,:,order);
               trainData.target = trainData.target(:,:,:,order);
            end

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

                fprintf('UL-tracker: incremental training: epoch %02d: %3d/%3d:', e, it, trainBatches) ;
            
                fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
                for f = setdiff(fieldnames(stats)', {'num', 'time', 'train'})
                    f = char(f) ;
                    fprintf(' %s: %.3f', f, stats.(f)) ;
                end
                fprintf('\n') ;
            end
        end
        
        saveShot(net, state, stats, modelPath(e));
    end
        
        
        
        
%         start = tic;
%         for it = startIter+1:opts.trainOpts.maxIters
%             % -------------------------------------------------------------
%             % tracking and selecting training pairs
%             % -------------------------------------------------------------
%             if mod(it, opts.trackOpts.trackPerIters) == 1 ...
%                 && it ~= opts.trainOpts.maxIters
%                 tracked = floor(it / opts.trackOpts.trackPerIters) + 1;
%                 fprintf('[stage 1] sst incremental training: epoch %02d: iter %5d / %5d track %d ...\n', e, it, opts.trainOpts.maxIters, tracked) ;
%                 
% %                 net_track = split_net(net, opts.trackOpts.layer4Tracking);
% %                 
% %                 if opts.gpus >= 1
% %                     net_track.move('gpu');
% %                     net.move('gpu');
% %                 end
% %                 
% %                 if strcmpi(opts.arch, 'DCFNet')
% %                     net_track.layers(1).block.pad = 1;
% %                     net_track.layers(3).block.pad = 1;  
% %                 end
%                 bstart = (tracked-1) * opts.trackOpts.clipSize + 1;
%                 bstart = mod(bstart, numel(perm));
%                 bend = min(bstart + opts.trackOpts.clipSize - 1, numel(imdb.images.data));
%                 trainData = opts.trainOpts.getDataOnlineFcn(imdb, net, perm(bstart:bend), opts.trackOpts);
%                 
%                 if opts.saveInternalPairs
%                     vaMakeDir(opts.outPairImgDir);
%                     fprintf('[stage 1] sst incremental training: epoch %02d: saving internal discovered pair images ...\n', e) ;
%                     target = uint8(gather(trainData.target));
%                     search = uint8(gather(trainData.search));
%                     save(fullfile(opts.outPairImgDir, sprintf('pair_epoch%02d_track%03d.mat', e, tracked)), 'target', 'search');
%                 end      
%                 
%                 if opts.trainOpts.lossSelection && ~isempty(opts.trainOpts.lossRange)
%                     trainData = select_sample_with_loss(trainData, net, opts.trainOpts);
%                 end
%             end            
%             
%             % -------------------------------------------------------------
%             % training
%             % -------------------------------------------------------------
%             numTrainImages = size(trainData.target, 4);
%             istart = mod(it, opts.trackOpts.trackPerIters);
%             if istart == 0
%                 istart = 1;
%             end
%             bstart = mod((istart - 1) * opts.trainOpts.batchSize + 1, numTrainImages);
%             if bstart == 0
%                 bstart = 1;
%             end
%             bend = min(bstart + opts.trainOpts.batchSize - 1, numTrainImages); 
%             batch = bstart:bend;
%             
%             if bstart <= 10
%                 order = randperm(numTrainImages);
%                 trainData.target = trainData.target(:,:,:,order);
%                 trainData.search = trainData.search(:,:,:,order);
%             end
%             
%             inputs = opts.trainOpts.getBatchFcn(trainData, batch);
%             net.mode = 'normal' ;
%             net.eval(inputs, opts.trainOpts.derOutputs) ;
%             state = accumulateGradients(net, state, opts.trainOpts, numel(batch), []) ;
%             
%             % Get statistics.
%             time = toc(start) ;
%             batchTime = time - stats.time ;
%             stats.num = stats.num + numel(batch);
%             stats.time = time ;
%             stats = extractStats(stats, net) ;
%             currentSpeed = numel(batch) / batchTime ;
%             averageSpeed = stats.num / time ;
% 
%             fprintf('[stage 1] sst incremental training: epoch %02d: %3d/%3d:', e, it, opts.trainOpts.maxIters) ;
%             
%             fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
%             for f = setdiff(fieldnames(stats)', {'num', 'time', 'train'})
%                 f = char(f) ;
%                 fprintf(' %s: %.3f', f, stats.(f)) ;
%             end
%             fprintf('\n') ;
%             
%             if mod(it, opts.trainOpts.snapshot) == 0
%                 state.stats.train = stats ;
%                 for i = 1:numel(state.solverState)
%                     s = state.solverState{i} ;
%                     if isnumeric(s)
%                         state.solverState{i} = gather(s) ;
%                     elseif isstruct(s)
%                         state.solverState{i} = structfun(@gather, s, 'UniformOutput', false) ;
%                     end
%                 end
%         
%                 net.reset() ;
%                 net.move('cpu') ;
%         
%                 saveState(modelPath(e, it), net, state) ;
%         
%                 stats.train = state.stats.train;
%                 saveStats(modelPath(e, it), stats);
%                 
%                 if opts.gpus >= 1
%                     net.move('gpu');
%                     for i = 1:numel(state.solverState)
%                         s = state.solverState{i} ;
%                         if isnumeric(s)
%                             state.solverState{i} = gpuArray(s) ;
%                         elseif isstruct(s)
%                             state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
%                         end
%                     end
%                 end
%             end
%         end
%         
%         % Save back to state.
%         state.stats.train = stats ;
%    
%         for i = 1:numel(state.solverState)
%             s = state.solverState{i} ;
%             if isnumeric(s)
%                 state.solverState{i} = gather(s) ;
%             elseif isstruct(s)
%                 state.solverState{i} = structfun(@gather, s, 'UniformOutput', false) ;
%             end
%         end
%         
%         net.reset() ;
%         net.move('cpu') ;
%         startIter = 0;
%         
%         saveState(modelPath(e, opts.trainOpts.maxIters), net, state) ;
%         
%         stats.train = state.stats.train;
%         saveStats(modelPath(e, opts.trainOpts.maxIters), stats);
%     end   
end

