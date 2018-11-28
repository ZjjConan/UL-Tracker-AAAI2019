function state = track_step(state, img) 
    state.currFrame = state.currFrame + 1;
    if state.opts.gpus
        img = gpuArray(img);
    end
    
    numTargets = size(state.targetSize, 1);
    
    state.window_sz = state.targetSize';
    state.window_sz = [state.window_sz * state.opts.scale_factor(1), ...
                       state.window_sz * state.opts.scale_factor(2), ...
                       state.window_sz * state.opts.scale_factor(3)];
    state.window_sz = state.window_sz * (1 + state.opts.padding);
    patch_crop = track_crop_impl(img, state.targetPosition, state.window_sz, ...
        state.opts.net_input_size, state.opts.yyxx);
    if isscalar(state.opts.net_average_image)
        search = patch_crop - state.opts.net_average_image;
    else
        search = bsxfun(@minus, patch_crop, state.opts.net_average_image);
    end
    state.net.eval({'target', search});

    z = bsxfun(@times, state.net.vars(end).value, state.cos_window);
    zf = fft2(z);
    kzf = squeeze(sum(zf .* repmat(conj(state.model_xf), [1 1 1 state.opts.num_scale]),3)/state.numel_xf);

    for s = 1:state.opts.num_scale
        bstart = (s - 1) * size(state.model_alphaf, 3) + 1;
        bend = s * size(state.model_alphaf, 3);
        response{s} = bsxfun(@times, state.model_alphaf, kzf(:,:,bstart:bend));
    end
    response = gather(squeeze(real(ifft2(cat(3, response{:})))));
    window_sz = zeros(2, numTargets);
    state.max_response = zeros(numTargets, 1);
    for i = 1:numTargets
        index = [i, i+numTargets, i+2*numTargets];
        [max_response, max_index] = max(reshape(response(:,:,index),[],state.opts.num_scale));
        [~,scale_delta] = max(max_response.*state.opts.scale_penalties);
        [vert_delta, horiz_delta] = ind2sub(state.opts.net_input_size, max_index(scale_delta));

        if vert_delta > state.opts.net_input_size(1) / 2  %wrap around to negative half-space of vertical axis
            vert_delta = vert_delta - state.opts.net_input_size(1);
        end
        if horiz_delta > state.opts.net_input_size(2) / 2  %same for horizontal axis
            horiz_delta = horiz_delta - state.opts.net_input_size(2);
        end 
        window_sz(:, i) = state.window_sz(:,index(scale_delta));
        state.targetPosition(i, :) = state.targetPosition(i, :) + [vert_delta - 1, horiz_delta - 1].*...
            window_sz(:, i)'./state.opts.net_input_size;
        state.targetSize(i, :) = min(max(...
                window_sz(:, i)'./(1+state.opts.padding), state.min_sz(i, :)), state.max_sz(i, :));
        state.max_response(i) = max(max_response);
    end
    state.window_sz = window_sz;
    state.results{state.currFrame} = [state.targetPosition(:,[2,1]) - state.targetSize(:,[2,1])/2, ...
        state.targetSize(:,[2,1])];
end

