function boxes = ulClipBox(boxes, imsize)
    boxes(:,3:4) = boxes(:,1:2) + boxes(:,3:4);
    boxes(:,1) = max(min(boxes(:,1), imsize(1)), 1);
    boxes(:,2) = max(min(boxes(:,2), imsize(2)), 1);
    boxes(:,3) = max(min(boxes(:,3), imsize(1)), 1);
    boxes(:,4) = max(min(boxes(:,4), imsize(2)), 1);
    boxes(:,3:4) = boxes(:,3:4) - boxes(:,1:2);
end