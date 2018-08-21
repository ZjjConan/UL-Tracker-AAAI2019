function data = cfnet_get_data_online(imdb, net, batch, opts)
   
    exemplar_size = 255;
    instance_size = 255;
    context_amount = 0.5;

    [images, bboxes] = opts.getBatchFcn(imdb, batch, opts);
    
    numImages = size(images, 4)/2;
    net.mode = 'test';
    
    data.target = cell(numImages, 1);
    data.search = cell(numImages, 1);

    tic
    for i = 1:numImages     
        imgs = images(:,:,:,[i, i+numImages]);
        bbox = bboxes{i}(1:min(opts.numSamples, size(bboxes{i}, 1)), 1:4);

        trajectory = opts.trackerFcn(net, imgs, bbox, opts);
        [score, order] = rankTrajectory(trajectory);
        sel = order(1:min(numel(order), opts.numSelects));
        x_boxes = trajectory.for{1}(sel, :);
        z_boxes = trajectory.for{2}(sel, :);
        
        for j = 1:size(x_boxes, 1)
            [im_crop_z, bbox_z, pad_z, im_crop_x, bbox_x, pad_x]  = get_crops(imgs(:,:,:,1), x_boxes, exemplar_size, instance_size, context_amount);
        end
        
%         x_pos = (x_boxes(:, 1:2) + x_boxes(:, 3:4) / 2)';
%         z_pos = (z_boxes(:, 1:2) + z_boxes(:, 3:4) / 2)';
%         x_sz  = (x_boxes(:, 3:4) * (1 + net.meta.padding))';
%         z_sz  = (z_boxes(:, 3:4) * (1 + net.meta.padding))';
%          
%         if opts.gpus >= 1
%             imgs = gpuArray(imgs);
%             x_pos = gpuArray(x_pos);
%             z_pos = gpuArray(z_pos);
%             x_sz = gpuArray(x_sz);
%             z_sz = gpuArray(z_sz);
%         end
%         
%         data.target{i} = ...
%             vaCropUsingBilinearSampler(imgs(:,:,:,1), x_pos, x_sz, net.meta.inputSize, opts.yyxx);
%         
%         data.search{i} = ...
%             vaCropUsingBilinearSampler(imgs(:,:,:,2), z_pos, z_sz, net.meta.inputSize, opts.yyxx);
        
        fprintf('train: mining %3d / %3d images with %3d boxes time %.2fs\n', ...
            i, numImages, opts.numSamples, toc);
    end
    
    data.target = cat(4, data.target{:});
    data.search = cat(4, data.search{:});
end


function [score, order] = rankTrajectory(trajectory)
    score = zeros(size(trajectory.for{1}, 1), 1);
    for t = 1:size(trajectory.for{1}, 1)
        score(t) = bboxOverlapRatio(trajectory.for{1}(t, :), trajectory.bak{1}(t, :));
    end
    [score, order] = sort(score, 'descend');
end

