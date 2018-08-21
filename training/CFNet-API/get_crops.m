function [im_crop_z, bbox_z, pad_z, im_crop_x, bbox_x, pad_x] = get_crops(im, bbox, size_z, size_x, context_amount)
    %% Get exemplar sample
    % take bbox with context for the exemplar

%     bbox = double(object.extent);
    [cx, cy, w, h] = deal(bbox(1)+bbox(3)/2, bbox(2)+bbox(4)/2, bbox(3), bbox(4));
    wc_z = w + context_amount*(w+h);
    hc_z = h + context_amount*(w+h);
    s_z = sqrt(single(wc_z*hc_z));
    scale_z = size_z / s_z;
    [im_crop_z, left_pad_z, top_pad_z, right_pad_z, bottom_pad_z] = get_subwindow_avg(im, [cy cx], [size_z size_z], [round(s_z) round(s_z)]);
    pad_z = ceil([scale_z*(left_pad_z+1) scale_z*(top_pad_z+1) size_z-scale_z*(right_pad_z+left_pad_z) size_z-scale_z*(top_pad_z+bottom_pad_z+1)]);
    %% Get instance sample
    d_search = (size_x - size_z)/2;
    pad = d_search/scale_z;
    s_x = s_z + 2*pad;
    scale_x = size_x / s_x;
    [im_crop_x, left_pad_x, top_pad_x, right_pad_x, bottom_pad_x] = get_subwindow_avg(im, [cy cx], [size_x size_x], [round(s_x) round(s_x)]);
    pad_x = ceil([scale_x*(left_pad_x+1) scale_x*(top_pad_x+1) size_x-scale_x*(right_pad_x+left_pad_x) size_x-scale_x*(top_pad_x+bottom_pad_x+1)]);
    % Size of object within the crops
    ws_z = w * scale_z;
    hs_z = h * scale_z;
    ws_x = w * scale_x;
    hs_x = h * scale_x;
    bbox_z = [(size_z-ws_z)/2, (size_z-hs_z)/2, ws_z, hs_z];
    bbox_x = [(size_x-ws_x)/2, (size_x-hs_x)/2, ws_x, hs_x];
end