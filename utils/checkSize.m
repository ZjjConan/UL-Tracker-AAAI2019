function ok = checkSize(frame_sz, object_extent, min_ratio, max_ratio)
% From SiameseFC
    object_extent = single(object_extent);
    if nargin < 3
        min_ratio = 0.1;
    end
    if nargin < 4
        max_ratio = 0.75;
    end
%     % accept only objects >10% and <75% of the total frame
    area_ratio = sqrt((object_extent(:, 3) .* object_extent(:, 4)) / prod(frame_sz));
    ok = area_ratio > min_ratio & area_ratio < max_ratio;
end