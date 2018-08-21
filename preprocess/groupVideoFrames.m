function groupVideoFrames(varargin)

    opts.imgDir = '';
    opts.boxDir = '';
    opts.debug = false;

    opts.saveDir = 'data';
    opts.useCorr = true;
    opts.batchSize = 500;
    opts.corrMinThre = 0.1;
    opts.resizeRatio = 0.3;
    
    [opts, varargin] = vl_argparse(opts, varargin);
    
    images = ulDir(opts.imgDir, 'jpg');
    im = ulReadImage(images{1});
    [r, c, ~] = size(im);
    imsz = round([r c] * opts.resizeRatio);
    numBatches = round(numel(images) / opts.batchSize);
    score = zeros(numel(images), 1);
    score(1) = 1;
    tic
    for b = 1:numBatches
        bstart = (b-1) * opts.batchSize + 1;
        bend = min(b * opts.batchSize + 1, numel(images));
        batch = bstart:bend;
        imgs = ulReadImages(images(batch), 4, false, imsz);
           
        imgs = cellfun(@rgb2gray, ...
                       cellfun(@uint8, imgs, 'UniformOutput', false), ...
                       'UniformOutput', false);
        
        for i = 1:numel(imgs)-1
            score(bstart+i) = corr2(imgs{i}, imgs{i+1});       
        end
        fprintf('%s: compute pixel correlation for batch [%d -> %d] / %d time %.2fs\n', ...
            mfilename, bstart, bend, numel(images), toc);
    end
    
    clips = zeros(numel(images), 2);
    clips(:, 2) = score;
    index = find(score <= opts.corrMinThre);
    index = [1; index];
    for i = 1:numel(index)-1
        clips(index(i):index(i+1)-1, 1) = i;
    end
    clips(index(i):end, :) = numel(index);
    clips(isnan(score)) = -1;
    ulMakeDir(fileparts(opts.saveDir));
    save(opts.saveDir, 'clips', '-v7.3'); 
end

