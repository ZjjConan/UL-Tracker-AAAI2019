function ims = ulReadImages(images, numThreads, isPack, imageSize)
    
    if nargin < 2, numThreads = 4; end
    if nargin < 3, isPack = false; end
    if nargin < 4, imageSize = []; end
   
    if isPack
        if isempty(imageSize)
            im = vl_imreadjpeg(images(1));
            [r, c] = size(im{1});
            imageSize = [r c];
        end
        
    	ims = vl_imreadjpeg(images, 'Pack', ...
    						'NumThreads', numThreads, ...
    					    'Resize', imageSize);
        ims = ims{1};
    else
    	ims = vl_imreadjpeg(images, 'NumThreads', numThreads, 'Resize', imageSize);
    end

end