function state = cfnet_state_initialize_batch(net, img, region, varargin)
    opts.join.method = 'corrfilt';
    opts.numScale = 3;
    opts.scaleStep = 1.0575;
    opts.scalePenalty = 0.9780;
    opts.scaleLR = 0.52;
    opts.responseUp = 8;
    opts.wInfluence = 0.2625; % influence of cosine window for displacement penalty
    opts.minSFactor = 0.2;
    opts.maxSFactor = 5;
    opts.zLR = 0.005; % update rate of the exemplar for the rolling avg (use very low values <0.015)
    opts.video = '';
    opts.visualization = false;
    opts.gpus = [];
    opts.track_lost = [];
    opts.startFrame = 1;
    opts.fout = -1;
    opts.imgFiles = [];
    opts.targetPosition = [];
    opts.targetSize = [];
    opts.track_lost = [];
    opts.ground_truth = [];
    
    opts.gpus = true;
    opts.scoreSize = 33;
    opts.totalStride = 4;
    opts.contextAmount = 0.5; % context amount for the exemplar

    opts.prefix_z = 'br1_'; % used to identify the layers of the exemplar
    opts.prefix_x = 'br2_'; % used to identify the layers of the instance
    opts.id_score = 'score';
    opts.trim_z_branch = {'br1_'};
    opts.trim_x_branch = {'br2_', 'join_xcorr', 'fin_adjust'};
    
    opts.gpu = true;
    opts.visualization = false;
    opts.averageImage = reshape(single([123,117,104]),[1,1,3]);
    
    [opts, varargin] = vl_argparse(opts, varargin);
    
    opts.subMean = false;
    
    % network surgeries depend on the architecture    
    switch opts.join.method
        case 'xcorr'
            opts.trim_x_branch = {'br2_','join_xcorr','fin_'};
            opts.trim_z_branch = {'br1_'};
			opts.exemplarSize = 127;
			opts.instanceSize = 255;
        case 'corrfilt'
            opts.trim_x_branch = {'br2_','join_xcorr','fin_adjust'};
            opts.trim_z_branch = {'br1_','join_cf','join_crop_z'};
			opts.exemplarSize = 255;
			opts.instanceSize = 255;
        otherwise
            error('network type unspecified');
    end

    
   state.targetPosition = region(:, [2,1]) + region(:, [4,3])/2;
   state.targetSize = region(:, [4,3]);

%    net_x = net;
   net_z = dagnn.DagNN.loadobj(net.copy());
    
   switch opts.join.method
       case 'xcorr'
           assert(~find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
       case 'corrfilt'
           assert(find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
   end
    
   net_x = dagnn.DagNN.loadobj(net.copy());

   net_z = init_net(net_z, [], false);
   net_x = init_net(net_x, [], false);
    
   for i=1:numel(opts.trim_x_branch)
       remove_layers_from_prefix(net_z, opts.trim_x_branch{i});
   end
    for i=1:numel(opts.trim_z_branch)
        remove_layers_from_prefix(net_x, opts.trim_z_branch{i});
    end
    
    state.net_z = net_z; clear net_z
    state.net_x = net_x; clear net_x
    
    if ~isempty(opts.gpus)
        state.net_z.move('gpu');
        state.net_x.move('gpu');
        net.move('gpu');
    end

    state.z_out_id = state.net_z.getOutputs();
    
    if ~isempty(opts.gpus)
        img = gpuArray(img);
    end
    
    if(size(img, 3)==1)
        img = repmat(img, [1 1 3]);
    end
    
    state.avgChans = gather([mean(mean(img(:,:,1))) mean(mean(img(:,:,2))) mean(mean(img(:,:,3)))]);

    wc_z = state.targetSize(:, 2) + opts.contextAmount*sum(state.targetSize, 2);
    hc_z = state.targetSize(:, 1) + opts.contextAmount*sum(state.targetSize, 2);
    state.s_z = sqrt(wc_z.*hc_z);
    state.s_x = opts.instanceSize/opts.exemplarSize * state.s_z;
    state.scales = (opts.scaleStep .^ ((ceil(opts.numScale/2)-opts.numScale) : floor(opts.numScale/2)));
    
    scaledExemplar = bsxfun(@times, state.s_z, state.scales);
    z_crop = cell(size(state.targetSize, 1), 1);
    for i = 1:size(state.targetSize, 1)
        [z_crop{i}, ~] = make_scale_pyramid(single(img), state.targetPosition(i, :), ...
            scaledExemplar(i, :), opts.exemplarSize, state.avgChans, [], opts);
        z_crop{i} = z_crop{i}(:,:,:,ceil(opts.numScale/2));
    end
    z_crop = cat(4, z_crop{:});

    if opts.averageImage
        z_crop = bsxfun(@minus, z_crop, opts.averageImage);
    end

    state.net_z.eval({'exemplar', z_crop});
    state.get_vars = @(net, ids) cellfun(@(id) net.getVar(id).value, ids, 'UniformOutput', false);
    state.z_out_val = state.get_vars(state.net_z, state.z_out_id);
    state.z_out_val = state.z_out_val{1};
    state.min_s_x = opts.minSFactor*state.s_x;
    state.max_s_x = opts.maxSFactor*state.s_x;
    state.min_s_z = opts.minSFactor*state.s_z;
    state.max_s_z = opts.maxSFactor*state.s_z;
  
    switch opts.join.method
        case 'corrfilt'
            opts.id_score = 'join_out';
            state.net_x.vars(end-1).precious = true;
    end
    
    window = single(hann(opts.scoreSize*opts.responseUp) * hann(opts.scoreSize*opts.responseUp)');
    state.window = window / sum(window(:));
    
    state.scoreId = state.net_x.getVarIndex(opts.id_score);
    state.opts = opts;
    state.currFrame = 1;
    state.results{state.currFrame} = [state.targetPosition(:,[2,1]) - state.targetSize(:,[2,1])/2, ...
        state.targetSize(:,[2,1])];
end
