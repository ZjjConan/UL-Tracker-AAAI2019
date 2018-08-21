function matches = DCFNetFBWTracking(net, imgs, boxes, opts)

    numImages = size(imgs, 4);

    if opts.gpus
        images = gpuArray(single(imgs));
    else
        images = single(imgs);
    end
    
    numBatches = ceil(size(boxes, 1) / opts.FBWBatchSize);
    for b = 1:numBatches
        bstart = (b-1) * opts.FBWBatchSize + 1;
        bend = min(b * opts.FBWBatchSize, size(boxes, 1));
        batch = bstart:bend;
        for i = 1:numImages
            if i == 1
                state = DCFNetInit(net, images(:,:,:,i), boxes(batch, :), opts);
            else
                state = DCFNetTrack(state, images(:,:,:,i));
            end
        end

        matches.for{b} = state.results;

        for i = numImages:-1:1
            if i == numImages
                state = DCFNetInit(net, images(:,:,:,numImages), matches.for{b}{end}, opts);
            else
                state = DCFNetTrack(state, images(:,:,:,i));
            end
        end

        matches.bak{b} = state.results(end:-1:1);
    end
     
    matches.for = cat(2, matches.for{:}); 
    matches.for{1} = cat(1, matches.for{1:2:end});
    matches.for{2} = cat(1, matches.for{2:2:end});
    matches.for(3:end) = [];
    
    matches.bak = cat(2, matches.bak{:}); 
    matches.bak{1} = cat(1, matches.bak{1:2:end});
    matches.bak{2} = cat(1, matches.bak{2:2:end});
    matches.bak(3:end) = [];
end

