function [imgs, boxes] = getBatchFromClip(imdb, batch, opts) 
    
    clips = imdb.images.data(batch);
    boxes = imdb.images.bbox(batch); 
    maxStep = imdb.images.maxStep(batch);
    
    currImages = cell(numel(clips), 1);
    currBBoxes = cell(numel(clips), 1);
    nextImages = cell(numel(clips), 1);
    nextBBoxes = cell(numel(clips), 1);
    for i = 1:numel(clips)
        nums = numel(clips{i});
        curr = randperm(nums, min(opts.numImagesPerClip, nums));
        next = randi([1 opts.maxInterval], 1, min(opts.numImagesPerClip, nums));
        next = curr + min(next, maxStep{i}(curr));
        currImages{i} = clips{i}(curr);
        nextImages{i} = clips{i}(next);
        currBBoxes{i} = boxes{i}(curr);
        nextBBoxes{i} = boxes{i}(next);
    end
    
    images = cat(1, currImages{:}, nextImages{:});
    boxes = cat(1, currBBoxes{:}, nextBBoxes{:});

    info = imfinfo(images{1});
    imgs = vl_imreadjpeg(images, 'NumThreads', 6, 'Pack', 'Resize', [info.Height, info.Width]);
    imgs = imgs{1};
end