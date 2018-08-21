function im = drawBox(im, boxes, color, hold)
% copied from Ross Girshick
% Fast R-CNN
% Copyright (c) 2015 Microsoft
% Licensed under The MIT License [see LICENSE for details]
% Written by Ross Girshick
% --------------------------------------------------------
% source: https://github.com/rbgirshick/fast-rcnn/blob/master/matlab/showboxes.m
%
% Fast R-CNN
% 
% Copyright (c) Microsoft Corporation
% 
% All rights reserved.
% 
% MIT License
% 
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the "Software"),
% to deal in the Software without restriction, including without limitation
% the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
% OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
% ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
% OTHER DEALINGS IN THE SOFTWARE.

    if nargin < 3
        randcolor = true;
    else
        randcolor = false;
    end
    
    if nargin < 4
        hold = false;
    end
    
    if size(boxes, 2) == 5
        drawScore = true;
    else
        drawScore = false;
    end
%     hold on;
    imshow(im); 
%     hold on;
%     set(gcf, 'Color', 'white');

    s = '-';
%     boxes(:, 3:4) = boxes(:, 1:2) + boxes(:, 3:4) - 1;

    if ~isempty(boxes)
%         x1 = boxes(:, 1);
%         y1 = boxes(:, 2);
%         x2 = boxes(:, 3);
%         y2 = boxes(:, 4);
%         line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', ...
%             'color', color, 'linewidth', 1.5, 'linestyle', s);
        for i = 1:size(boxes, 1)
            if randcolor
                rectangle('Position', boxes(i, 1:4), 'LineWidth', 4, ...
                    'LineStyle', s, 'EdgeColor', rand(1,3));
            else
                rectangle('Position', boxes(i, 1:4), 'LineWidth', 4, ...
                    'LineStyle', s, 'EdgeColor', color);
            end
            if drawScore
                text(double(boxes(i, 1)), double(boxes(i, 2)) - 2, ...
                     sprintf('%.3f', boxes(i, 5)), ...
                     'backgroundcolor', 'w', 'color', 'k');
            end
        end
    end
    
    if hold
        im = getframe();
        im = im.cdata;
    end
end