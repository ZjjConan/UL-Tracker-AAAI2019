function crop = ulCrop(im, box)
    crop = im(box(2):box(2)+box(4)-1, box(1):box(1)+box(3)-1, :);
end

