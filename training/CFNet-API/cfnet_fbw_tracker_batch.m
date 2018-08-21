function res = cfnet_fbw_tracker_batch(net, imgs, boxes, opts)

    numImages = size(imgs, 4);

    if opts.gpus
        imgsOnDevice = gpuArray(single(imgs));
    else
        imgsOnDevice = single(imgs);
    end

    boxes(:, 1:2) = boxes(:, 1:2) - 1;
    numBatches = max(1, round(size(boxes, 1) / opts.FBWDSize));
    for b = 1:numBatches
        bstart = (b-1) * opts.FBWDSize + 1;
        bend = min(b * opts.FBWDSize, size(boxes, 1));
        batch = bstart:bend;
        for i = 1:numImages
            if i == 1
                state = cfnet_state_initialize_batch(net, imgsOnDevice(:,:,:,i), boxes(batch, :), opts);
            else
                state = cfnet_tracker_batch(state, imgsOnDevice(:,:,:,i));
            end
        end

        res.for{b} = state.results;

        for i = numImages:-1:1
            if i == numImages
                state = cfnet_state_initialize_batch(net, imgsOnDevice(:,:,:,numImages), res.for{b}{end}, opts);
            else
                state = cfnet_tracker_batch(state, imgsOnDevice(:,:,:,i));
            end
        end

        res.bak{b} = state.results(end:-1:1);
    end
     
    res.for = cat(2, res.for{:}); 
    res.for{1} = cat(1, res.for{1:2:end});
    res.for{2} = cat(1, res.for{2:2:end});
    res.for(3:end) = [];
    
    res.bak = cat(2, res.bak{:}); 
    res.bak{1} = cat(1, res.bak{1:2:end});
    res.bak{2} = cat(1, res.bak{2:2:end});
    res.bak(3:end) = [];
end

