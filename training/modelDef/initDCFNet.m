function net = initDCFNet(opts)

    net = dagnn.DagNN() ;
    
    net.meta.normalization.imageSize = [opts.inputSize([1,1]),3];
    net.meta.normalization.averageImage = reshape(single([123,117,104]),[1,1,3]);
    net.meta.arch = 'DCFNet';
    net.meta.lossType = opts.lossType;
    net.meta.inputSize = opts.inputSize;
    net.meta.padding = opts.padding;

    conv1 = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1', conv1, {'target'}, {'conv1'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'conv1x'});
    
    conv2 = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2', conv2, {'conv1x'}, {'conv2'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2'}, {'conv2n'});
    
    net.addLayer('drop1', dagnn.DropOut, {'conv2n'}, {'x'});
    
    %% search
    conv1s = dagnn.Conv('size', [3 3 3 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv1s', conv1s, {'search'}, {'conv1s'}, {'conv1f', 'conv1b'}) ;
    net.addLayer('relu1s', dagnn.ReLU(), {'conv1s'}, {'conv1sx'});
    
    conv2s = dagnn.Conv('size', [3 3 32 32], 'pad', 0, 'stride', 1, 'dilate', 1, 'hasBias', true) ;
    net.addLayer('conv2s', conv2s, {'conv1sx'}, {'conv2s'}, {'conv2f', 'conv2b'}) ;
    net.addLayer('norm1s', dagnn.LRN('param',[5 1 0.0001/5 0.75]), {'conv2s'}, {'conv2sn'});
    
    net.addLayer('drop1s', dagnn.DropOut, {'conv2sn'}, {'z'});
    
    feature_sz = opts.inputSize([1,1]) - [4, 4];

    
    net.meta.interp_factor = 0.0110;
    net.meta.scale_penalty = 1;
    net.meta.scale_step = 1.0150;
    
    
    target_sz = opts.inputSize([1,1])/(1+opts.padding);
    sigma = sqrt(prod(target_sz))/10;
    DCF = dagnn.DCF('win_size', feature_sz,'sigma', sigma) ;
    net.addLayer('DCF', DCF, {'x','z'}, {'response'}) ;

    ResponseLossL2 = dagnn.ResponseLossL2('win_size', feature_sz, 'sigma', sigma) ;
    net.addLayer('ResponseLoss', ResponseLossL2, {'response'}, {'objective'}) ;

    ResponseLossL2Instance = dagnn.ResponseLossL2Instance('win_size', feature_sz, 'sigma', sigma);
    net.addLayer('ResponseLossInstance', ResponseLossL2Instance, {'response'}, {'objective_instance'});
    
    net.initParams();
end

