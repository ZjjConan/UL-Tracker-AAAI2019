function state = cfnet_tracker_batch(state, img)
    state.currFrame = state.currFrame + 1;
    nums = size(state.targetSize, 1);
    if ~isempty(state.opts.gpus)
        img = gpuArray(img);
        newTargetPosition = gpuArray.zeros(nums, 2);
        newScale = gpuArray.zeros(nums, 1);
    else
        newTargetPosition = zeros(nums, 2);
        newScale = zeros(nums, 1);
    end
    
    if(size(img, 3)==1), img = repmat(img, [1 1 3]); end
    scaledInstance = bsxfun(@minus, state.s_x, state.scales);
    
    copy = @(v, n) cellfun(@(x) repmat(x, [1 1 1 n]), v, 'UniformOutput', false);
    for i = 1:nums
        [x_crops, pad_masks_x] = make_scale_pyramid(img, state.targetPosition(i, :), scaledInstance(i, :), state.opts.instanceSize, ...
            state.avgChans, [], state.opts);
        z_out = interleave(state.z_out_id, copy({state.z_out_val(:,:,:,i)}, state.opts.numScale));
        [newTargetPosition(i, :), newScale(i, :)] = tracker_step(state.net_x, state.s_x(i), state.s_z(i), state.scoreId, ...
            z_out, x_crops, pad_masks_x, state.targetPosition(i, :), state.window, state.opts);
    end
%     s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));

    % update target position
    state.targetPosition = gather(newTargetPosition);
    scaledTarget = [state.targetSize(:, 1) .* state.scales; state.targetSize(:, 2) .* state.scales];
    for i = 1:nums
        state.targetSize(i, :) = (1-state.opts.scaleLR)*state.targetSize(i, :) + ...
            state.opts.scaleLR*[scaledTarget(i, newScale(i)) scaledTarget(i+nums, newScale(i))];
    end
    
    state.results{state.currFrame} = [state.targetPosition(:,[2,1]) - state.targetSize(:,[2,1])/2, ...
        state.targetSize(:,[2,1])];
end
