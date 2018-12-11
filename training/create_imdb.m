function imdb = create_imdb(varargin)
    
    opts.imgDir = '';
    opts.boxDir = '';
    opts.imdbPath = '';
    % segment video to clips
    opts.isSegment = true;
    opts.clipDir = '';
    opts.numProposals = 100;
    opts.scaleRatio = 1;
    opts.minFramesPerSegment = 10;
      
    [opts, ~] = vl_argparse(opts, varargin);
    
    imdb.images.data = ul_dir(opts.imgDir, 'jpg');
    boxFiles = ul_dir(opts.boxDir, 'mat');
    nbox = numel(boxFiles);
    bbox = cell(numel(boxFiles), 1);
    tic
    for i = 1:numel(boxFiles)
        s = load(boxFiles{i});
        bbox{i} = single(s.bbox(1:min(size(s.bbox,1),opts.numProposals), :) * opts.scaleRatio);
        if mod(i, 1000) == 0
            fprintf('%s: load %d / %d box file time %.2fs\n', mfilename, i, nbox, toc);
        end
    end
    imdb.images.bbox = bbox;
    
    if opts.isSegment
        if isempty(opts.clipDir)
            warning('no clip index file found, do not generate clips !!!')
        else
            tmpl = load(opts.clipDir);
            notok = tmpl.clips(:, 1) == -1; 
            data = imdb.images.data(~notok);
            bbox = imdb.images.bbox(~notok);
            tmpl.clips(notok, :) = [];
            clipIndex = tmpl.clips(:, 1);
            clip = unique(clipIndex);
            imdb.images.data = cell(numel(clip), 1);
            imdb.images.bbox = cell(numel(clip), 1);
            imdb.images.maxStep = cell(numel(clip), 1); 
            for c = 1:numel(clip)
                index = clipIndex == clip(c);
                if sum(index) <= opts.minFramesPerSegment
                    continue;
                    imdb.images.data{c}(notok) = [];
                    imdb.images.bbox{c}(notok) = [];
                end
                imdb.images.data{c} = data(index);
                imdb.images.bbox{c} = bbox(index);
                notok = cellfun(@isempty, imdb.images.bbox{c}, 'UniformOutput', true);
                imdb.images.data{c}(notok) = [];
                imdb.images.bbox{c}(notok) = [];
                imdb.images.maxStep{c} = numel(imdb.images.data{c})-1:-1:0;
            end
            notok = cellfun(@isempty, imdb.images.bbox, 'UniformOutput', true);
            imdb.images.data(notok) = [];
            imdb.images.bbox(notok) = [];
            imdb.images.maxStep(notok) = [];
        end
    else
        % remove empty box
        notok = cellfun(@isempty, imdb.images.bbox, 'UniformOutput', true);
        imdb.images.bbox(notok) = [];
        imdb.images.data(notok) = [];
    end
    
    imdb.meta.normalization.averageImage =  reshape(single([123,117,104]),[1,1,3]);
    
    ul_make_dir(fileparts(opts.imdbPath));
    save(opts.imdbPath, '-struct', 'imdb', '-v7.3'); 
end

