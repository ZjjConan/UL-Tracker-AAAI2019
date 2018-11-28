function state = track_init(net, img, region, varargin)

    opts.gpus = true;
    opts.visualization = false;

    opts.lambda = 1e-4;
    opts.padding = 2.0;
    opts.output_sigma_factor = 0.1;
    opts.interp_factor = 0.01;

    opts.num_scale = 3;
    opts.scale_step = 1.0275;
    opts.min_scale_factor = 0.2;
    opts.max_scale_factor = 5;
    opts.scale_penalty = 0.9925;
    opts.yyxx = [];
    [opts, varargin] = vl_argparse(opts, varargin);
    
    state.net = net;
    state.net.mode = 'test';
    
    state.opts = opts;
    
    state.opts.scale_factor = state.opts.scale_step.^((1:state.opts.num_scale) - ceil(state.opts.num_scale/2));
    state.opts.scale_penalties = ones(1, state.opts.num_scale);
    state.opts.scale_penalties((1:state.opts.num_scale) ~= ceil(state.opts.num_scale/2)) = state.opts.scale_penalty;

    state.opts.net_input_size = state.net.meta.normalization.imageSize(1:2);
    state.opts.net_average_image = state.net.meta.normalization.averageImage;

    output_sigma = sqrt(prod(state.opts.net_input_size./(1+state.opts.padding)))*state.opts.output_sigma_factor;
    
    output_sz = state.net.getVarSizes({'target', state.opts.net_input_size});
    state.yf = single(fft2(create_gaussian_labels(output_sigma, output_sz{end}(1:2))));
    state.cos_window = single(hann(size(state.yf,1)) * hann(size(state.yf,2))');

    if state.opts.gpus %gpuSupport
        state.opts.yyxx = gpuArray(state.opts.yyxx);
        state.net.move('gpu');
        img = gpuArray(single(img));
        state.yf = gpuArray(state.yf);
        state.cos_window = gpuArray(state.cos_window);
    end
    
    state.targetPosition = region(:, [2,1]) + region(:, [4,3])/2;
    state.targetSize = region(:, [4,3]);
    state.min_sz = max(4, state.opts.min_scale_factor .* state.targetSize);
    [im_h,im_w,~] = size(img);
    state.max_sz = min([im_h,im_w], state.opts.max_scale_factor .* state.targetSize);

    state.window_sz = state.targetSize'*(1+state.opts.padding);
    patch = track_crop_impl(img, state.targetPosition, state.window_sz, ...
        state.opts.net_input_size, state.opts.yyxx);
    
    if isscalar(state.opts.net_average_image)
        target = patch - state.opts.net_average_image; 
    else
        target = bsxfun(@minus, patch, state.opts.net_average_image);
    end

    state.net.eval({'target', target});

    x = bsxfun(@times, state.net.vars(end).value, state.cos_window);
    xf = fft2(x);
    state.numel_xf = numel(xf);
    kf = squeeze(sum(xf.*conj(xf),3)/state.numel_xf);
    state.model_alphaf = bsxfun(@rdivide, state.yf, kf + state.opts.lambda);
    state.model_xf = xf;
    state.results{1} = region;
    state.currFrame = 1;
end

