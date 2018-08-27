function images = augImages(images, varargin)
%AUGIMAGES 此处显示有关此函数的摘要
%   此处显示详细说明

    opts.blurImage = false;
    opts.blurSigma = 2;
    opts.blurProb = 0.25;
    
    opts.grayImage = false;
    opts.grayProb = 0.25;
    [opts, varargin] = vl_argparse(opts, varargin);
    
    numImages = size(images, 4) / 2;
    
    if opts.grayImage
        index = randperm(numImages, round(numImages * opts.grayProb));
        for i = 1:numel(index)
            images(:,:,:,index(i)) = ...
                repmat(rgb2gray(uint8(images(:,:,:,index(i)))), [1 1 3]);  
            images(:,:,:,index(i)+numImages) = ...
                repmat(rgb2gray(uint8(images(:,:,:,index(i)+numImages))), [1 1 3]);    
        end
        images = single(images);
    end
    
    if opts.blurImage
        index = randperm(numImages, round(numImages * opts.blurProb));
        sigma = rand(1, numel(index)) * opts.blurSigma;
        for i = 1:numel(index)
            images(:,:,:,index(i)) = ...
                imgaussfilt(images(:,:,:,index(i)), sigma(i));  
            images(:,:,:,index(i)+numImages) = ...
                imgaussfilt(images(:,:,:,index(i)+numImages), sigma(i));    
        end
    end
end

