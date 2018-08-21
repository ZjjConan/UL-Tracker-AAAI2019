function extractEdgeBoxes(varargin)
    opts.frameDir = '';
    opts.saveDir = '';
    opts.debug = false;
    
    % edge box params
    opts.modelPath = '/models/forest/modelBsds.mat';
    opts.ebOpts = edgeBoxes;
    opts.ebOpts.alpha = .65;     % step size of sliding window search
    opts.ebOpts.beta  = .1;     % nms threshold for object proposals
    opts.ebOpts.minScore = .01;  % min score of boxes to detect
    opts.ebOpts.maxBoxes = 1000;  % m
    
    [opts, varargin] = vl_argparse(opts, varargin);
    
    % load model
    model = load(opts.modelPath);
    model = model.model;
    model.opts.multiscale = 0; 
    model.opts.sharpen = 2; 
    model.opts.nThreads = 4;
    
    [images, names] = vaDir(opts.frameDir, 'jpg');
    
    vaMakeDir(opts.saveDir);
    ebOpts = opts.ebOpts;
    numImages = numel(images);
    saveDir = opts.saveDir;
%     tic
    parfor i = 1:numImages
%         if exist(fullfile(saveDir, [names{i} '.mat']))
%             fprintf('%s: process %d / %d frame time %.2fs\n', mfilename, i, numImages, toc);
%             continue;
%         end
        im = vaReadImage(images{i});
        bbox = edgeBoxes(im, model, ebOpts);
  
        parSaveBBox(fullfile(saveDir, [names{i} '.mat']), bbox);
        
        % debug
%         if opts.debug
%             imshow(im);
%             for p = 1:size(proposals, 1)
%                 rectangle('Position', proposals(p, 1:4), 'EdgeColor', rand(1,1,3), 'LineWidth', 1.5);
%             end
%         end
%         
        fprintf('%s: process %d / %d frame\n', mfilename, i, numImages);
    end
end

