function ul_group_frames(varargin)

    opts.imgDir = '';
    opts.boxDir = '';
    opts.debug = false;

    opts.saveDir = 'data';
    opts.useCorr = true;
    opts.batchSize = 500;
    opts.corrMinThre = 0.1;
    opts.resizeRatio = 0.3;
    
    [opts, varargin] = vl_argparse(opts, varargin);
    
    images = ul_dir(opts.imgDir, 'jpg');
    im = ul_read_img(images{1});
    [r, c, ~] = size(im);
    imsz = round([r c] * opts.resizeRatio);
    score = zeros(numel(images), 1);
    score(1) = 1;
    prevImg = ul_read_img(images{1}, 'gray');
    prevImg = imresize(prevImg, imsz);
    tic
    for i = 2:numel(images)
        currImg = ul_read_img(images{i}, 'gray');
        currImg = imresize(currImg, imsz);
        score(i) = corr2(prevImg, currImg);  
        prevImg = currImg;
        if mod(i, 100) == 0
            fprintf('%s: compute pixel correlation for image [%d -- %d] / %d time %.2fs\n', ...
                mfilename, i-1, i+1, numel(images), toc);
        end
    end
        
    clips = zeros(numel(images), 2);
    clips(:, 2) = score;
    index = find(score <= opts.corrMinThre);
    index = [1; index];
    for i = 1:numel(index)-1
        clips(index(i):index(i+1)-1, 1) = i;
    end
    clips(index(i):end, 1) = numel(index);
    clips(isnan(score), 1) = -1;
    ul_make_dir(fileparts(opts.saveDir));
    save(opts.saveDir, 'clips', '-v7.3'); 
end

