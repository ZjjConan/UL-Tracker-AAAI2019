function patches = cropImagePatches(im, box)
    patches = cell(size(box, 1), 1); 
    for b = 1:size(box, 1)
        patches{b} = ulCrop(im, box(b, 1:4));
    end
end