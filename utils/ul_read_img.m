function im = ul_read_img(imfile, mode)
    if nargin < 2
        im = imread(imfile);
        if size(im, 3) == 1
            im = repmat(im, [1 1 3]);
        end
    elseif strcmpi(mode, 'gray')
        im = imread(imfile);
        im = rgb2gray(im);
    end
end

