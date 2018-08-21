function windows = vaCropUsingBilinearSampler(img, pos, cropSize, outputSize, yyxx)
    
    if nargin < 5
        yi = linspace(-1, 1, outputSize(1));
        xi = linspace(-1, 1, outputSize(1));
        [xx, yy] = meshgrid(xi, yi);
        yyxx = single([yy(:), xx(:)]') ; % 2xM
    end

    % bbox [x y w h]
    img = single(img);
    [im_h, im_w, ~, ~] = size(img);
    
%     scale = cropSize/(cropSize - padding*2);
    
    n = size(pos, 2);
%     pos = (bbox(:, 1:2) + bbox(:, 3:4) / 2)';
%     sz = bbox(:, 3:4)' * scale;

    im_h = im_h - 1;
    im_w = im_w - 1;

    cy_t = pos(2,:) * 2 / im_h - 1;
    cx_t = pos(1,:) * 2 / im_w - 1;

    h_s = cropSize(2,:) / im_h;
    w_s = cropSize(1,:) / im_w;

    s = reshape([h_s;w_s], 2,1, n); % x,y scaling
    t = [cy_t; cx_t]; % translation
    t = reshape(t, 2, 1, size(t, 2));
    g = bsxfun(@times, yyxx, s); % scale
    g = bsxfun(@plus, g, t); % translate
    g = reshape(g, 2, outputSize, outputSize, n);

    windows = vl_nnbilinearsampler(img, g);
end
