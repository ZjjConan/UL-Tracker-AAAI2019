function img_crop = track_crop_impl(img, pos, sz, output_sz, yyxx)
    [im_h,im_w,im_c,~] = size(img);

    if im_c == 1
        img = repmat(img,[1,1,3,1]);
    end

    im_h = im_h - 1;
    im_w = im_w - 1;

    cy_t = (pos(:,1)*2/im_h)-1;
    cx_t = (pos(:,2)*2/im_w)-1;

    h_s = sz(1,:)/im_h;
    w_s = sz(2,:)/im_w;

    s = reshape([h_s;w_s], 2,1,[]); % x,y scaling
    t = [cy_t,cx_t]'; % translation
    t = reshape(t, size(t,1), 1, size(t,2));
    if size(t, 3) ~= size(s, 3)
        t = repmat(t, [1 1 size(s,3)/size(t,3)]); 
    end
     
    g = bsxfun(@times, yyxx, s); % scale
    g = bsxfun(@plus, g, t); % translate
    g = reshape(g, 2, output_sz(1), output_sz(2), []);

    img_crop = vl_nnbilinearsampler(img, g);
end