function ulRemoveBoxes(varargin)

    opts.movieName = '';
    opts.imgDir = '';
    opts.boxDir = '';
    opts.saveDir = '';
    opts.debug = false;
    
    opts.removeWithMeanIntensity = false;
    opts.intensityRange = [];

    opts.removeWithRatio = false;
    opts.boxRatio = [];
    
    opts.removeWithBorder = false;
    
    opts.removeWithNMS = false;
    opts.NMSThreshold = 0.3;
    
    [opts, varargin] = vl_argparse(opts, varargin);
    
    ulMakeDir(opts.saveDir);
    
    if opts.removeWithMeanIntensity && isempty(opts.intensityRange)
        opts.intensityRange = [50, 200];
    end
    
    if opts.removeWithRatio && isempty(opts.boxRatio)
        opts.boxRatio = [0.1 0.7];
    end
    
    images = ulDir(opts.imgDir, 'jpg');
    [boxes, names] = ulDir(opts.boxDir, 'mat');
    iminfo = imfinfo(images{1});
    frameSize = [iminfo.Width, iminfo.Height];
    tic
    for i = 1:numel(images)
        if exist(fullfile(opts.saveDir, [names{i} '.mat']))
            continue;
        end
        
        s = load(boxes{i});
        bbox = s.bbox;
        img = ulReadImage(images{i}, 'gray');
        
        if opts.removeWithRatio
            ok = checkSize(frameSize, bbox, opts.boxRatio(1), opts.boxRatio(2));
        end
        
        if opts.removeWithBorder
            ok = ok | checkBorder(frameSize, bbox);
        end
        
        bbox(~ok, :) = [];
        
        if ~isempty(bbox)
            bbox = ulClipBox(bbox, frameSize);
            if opts.removeWithMeanIntensity
                patches = cropImagePatches(img, bbox);
                mu = cellfun(@mean, ...
                             cellfun(@reshape, ...
                                     cellfun(@single, patches, 'UniformOutput', false), ...
                                     repmat({1}, numel(patches), 1), repmat({[]}, numel(patches), 1), ...
                                     'UniformOutput', false ...
                                     ) ...
                             );
                ok = mu >= opts.intensityRange(1) & mu <= opts.intensityRange(2);
                bbox(~ok, :) = [];
            end
        end
        
        if opts.removeWithNMS
            bbox_ = bbox;
            bbox_(:,3:4) = bbox_(:,3:4) + bbox(:,1:2);
            pick = bboxNMS([bbox_ [size(bbox_,1):-1:1]'], opts.NMSThreshold);
            bbox = bbox(pick, :);
        end
        
        save(fullfile(opts.saveDir, [names{i} '.mat']), 'bbox');
        if mod(i, 100) == 0
            fprintf('%s: remove boxes in %d / %d image time %.2fs\n', ...
                mfilename, i, numel(images), toc);
        end
    end
end

