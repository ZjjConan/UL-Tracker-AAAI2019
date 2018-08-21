function data = DCFNetGetData(imdb, net, batch, opts)

    net4track = splitNet(net, opts.trackingFeatrLayer);                 
    net4track.layers(1).block.pad = 1;
    net4track.layers(3).block.pad = 1;  
    net4track.mode = 'test';
    
    opts.yyxx = initBilinearGrids(net.meta.inputSize);
    if opts.gpus >= 1
        net4track.move('gpu');
        net.move('gpu');
        opts.yyxx = gpuArray(opts.yyxx);
    end
 
    subBatchSize = 20;
    numBatches = ceil(numel(batch)/subBatchSize);
    data.target = cell(numBatches, 1);
    data.search = cell(numBatches, 1);
    inputSize = [net.meta.inputSize net.meta.inputSize];
    for b = 1:numBatches
        subBatchStart = (b - 1) * subBatchSize + 1;
        subBatchEnd = min(b * subBatchSize, numel(batch));
        [images, bboxes] = opts.getBatchFcn(imdb, batch(subBatchStart:subBatchEnd), opts);
        numImages = size(images, 4)/2;
        target = cell(numImages, 1);
        search = cell(numImages, 1);
        tic
        for i = 1:numImages
            imgs = images(:,:,:,[i, i+numImages]);
            bbox = single(bboxes{i});
            bbox = bbox(1:min(size(bbox,1),100), :);
          
            bbox(:, 1:2) = bbox(:, 1:2) - 1;
            matches = opts.trackingFcn(net4track, imgs, bbox, opts);    
            % FB verification
            score = FBWLocVerify(matches);
            % nms removing
            pick = bboxNMS([bbox score], 0);
            score = score(pick);
            % sort
            [score, order] = sort(score, 'descend');
            idx = score > opts.selectThre;
            order = order(idx);
            % select
            sel = order(1:min(numel(order), opts.selectNums));
            
            x_boxes = matches.for{1}(pick(sel), :);
%             y_boxes = matches.bak{1}(pick(sel), :);
            z_boxes = matches.for{2}(pick(sel), :);
            
%             for j = 1:size(x_boxes, 1)
%                 figure(1); imshow(uint8(imgs(:,:,:,1))); 
%                 rectangle('Position', x_boxes(j, :), 'EdgeColor', 'g');
%                 rectangle('Position', y_boxes(j, :), 'EdgeColor', 'r');
%                 figure(2); imshow(uint8(imgs(:,:,:,2))); 
%                 rectangle('Position', z_boxes(j, :), 'EdgeColor', 'g');
%             end
        
            x_pos = (x_boxes(:, 1:2) + x_boxes(:, 3:4) / 2);
            z_pos = (z_boxes(:, 1:2) + z_boxes(:, 3:4) / 2);
            x_sz  = (x_boxes(:, 3:4) * (1 + net4track.meta.padding))';
            z_sz  = (z_boxes(:, 3:4) * (1 + net4track.meta.padding))';
         
            if opts.gpus >= 1
                imgs = gpuArray(imgs);
                x_pos = gpuArray(x_pos);
                z_pos = gpuArray(z_pos);
                x_sz = gpuArray(x_sz);
                z_sz = gpuArray(z_sz);
            end
        
            target{i} = bilinearCrop(imgs(:,:,:,1), x_pos(:, [2,1]), x_sz([2,1], :), inputSize, opts.yyxx);
            search{i} = bilinearCrop(imgs(:,:,:,2), z_pos(:, [2,1]), z_sz([2,1], :), inputSize, opts.yyxx);
        
            fprintf('tracking %3d / %3d batch %3d / %3d images time %.2fs\n', b, numBatches, i, numImages, toc);
        end
    
        data.target{b} = gather(cat(4, target{:}));
        data.search{b} = gather(cat(4, search{:}));
    end
    data.target = cat(4, data.target{:});
    data.search = cat(4, data.search{:});
end


function score = FBWLocVerify(matches)
    score = diag(bboxOverlapRatio(matches.for{1}, matches.bak{1}));
end

function pick = NMSVerify(bbox)
    pick = bboxNMS(bbox);
end

