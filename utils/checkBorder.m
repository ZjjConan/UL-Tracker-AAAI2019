function ok = checkBorder(frame_sz, object_extent)
% From SiameseFC
    object_extent = single(object_extent);
    dist_from_border = 0.05 * (object_extent(:, 3) + object_extent(:, 4))/2;
    ok = object_extent(:, 1) > dist_from_border & object_extent(:, 2) > dist_from_border & ...
        (frame_sz(:, 1)-(object_extent(:, 1)+object_extent(:, 3))) > dist_from_border & ...
        (frame_sz(:, 2)-(object_extent(:, 2)+object_extent(:, 4))) > dist_from_border;
end