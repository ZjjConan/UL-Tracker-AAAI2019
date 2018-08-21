function bbox = bboxClip(bbox, imsz)

    bbox(:, 3:4) = bbox(:, 3:4) + bbox(:, 1:2);

    bbox(:, 1) = max(min(bbox(:, 1), imsz(2)), 1);
    bbox(:, 2) = max(min(bbox(:, 2), imsz(1)), 1);
    bbox(:, 3) = max(min(bbox(:, 3), imsz(2)), 1);
    bbox(:, 4) = max(min(bbox(:, 4), imsz(1)), 1);
    
    bbox(:, 3:4) = bbox(:, 3:4) - bbox(:, 1:2);
end
