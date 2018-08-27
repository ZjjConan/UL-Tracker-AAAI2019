function inputs = getTrainBatch(imdb, batch, varargin)
    opts.gpus = [];
    opts.averageImage = [];
    opts.augFlip = false;
    opts.augRotate = false;
    opts.rotateProb = 0.25;
    opts.rotateRange = [-60 60];
    
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
            inputs = {'target', target, 'search', fliplr(search)};
        end
    end
    
    if opts.augRotate
        index = randperm(size(target,4), round(size(target,4) * opts.rotateProb));
        theta = randi(opts.rotateRange, 1, numel(index));
        for i = 1:numel(index)
            search(:,:,:,index(i)) = imrotate(search(:,:,:,index(i)), theta(i), 'bilinear', 'crop');
        end
        inputs = {'target', target, 'search', search};
    end
end