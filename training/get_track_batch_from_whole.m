function [imgs, boxes] = get_track_batch_from_whole(imdb, batch, opts) 

    curr = batch;
    next = batch + randi(opts.maxInterval, 1, numel(batch));
    next = min(next, numel(imdb.images.data));
    
    boxes = imdb.images.bbox(curr);
    %     nextBoxes = imdb.images.bbox(nextFrameIndex);
    images = imdb.images.data([curr, next]);
    info = imfinfo(images{1});
    imgs = vl_imreadjpeg(images, 'NumThreads', 4, 'Pack', ...
        'Resize', [info.Height, info.Width]);
    imgs = imgs{1};
end