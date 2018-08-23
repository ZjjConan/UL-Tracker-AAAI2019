function inputs = getTrainBatch(imdb, batch, varargin)
    opts.gpus = [];
    opts.averageImage = [];
    opts.augFlip = false;
    opts.augGray = false;
    
    [opts, varargin] = vl_argparse(opts, varargin);
    
    target = single(imdb.target(:,:,:,batch));
    search = single(imdb.search(:,:,:,batch));
    
    if opts.gpus
        target = gpuArray(target);
        search = gpuArray(search);
    end
    
    if opts.augGray
        if rand > 0.75
            for i = 1:size(target, 4)
                target(:,:,:,i) = repmat(rgb2gray(uint8(target(:,:,:,i))), [1 1 3]);
                search(:,:,:,i) = repmat(rgb2gray(uint8(search(:,:,:,i))), [1 1 3]);
            end
            target = single(target);
            search = single(search);
        end
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
            inputs = {'target', target, 'search', fliplr(search)};
        end
    end
end