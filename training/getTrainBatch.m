function inputs = getTrainBatch(imdb, batch, varargin)
    opts.gpus = [];
    opts.averageImage = [];
    opts.augFlip = false;
    
    [opts, varargin] = vl_argparse(opts, varargin);
    
    target = single(imdb.target(:,:,:,batch));
    search = single(imdb.search(:,:,:,batch));
    
    if opts.gpus
        target = gpuArray(target);
        search = gpuArray(search);
    end
    
    if ~isempty(opts.averageImage)
        if isscalar(opts.averageImage)
            target = target - opts.averageImage;
            search = search - opts.averageImage;
        else
            target = bsxfun(@minus, target, opts.averageImage);
            search = bsxfun(@minus, search, opts.averageImage);
        end
    end
    
    inputs = {'target', target, 'search', search} ;
    
    if opts.augFlip
        if rand > 0.5
            inputs = {'target', fliplr(target), 'search', search};
        end
    end
end