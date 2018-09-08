clc; clear all; close all;
imdb = load('data/imdb/SR (1994)_imdb.mat');
% clipIndex = 6;
% 
% clips = imdb.images.data{clipIndex};
% for i = 1:50:numel(clips)
%     imshow(ulReadImage(clips{i}));
%     drawnow
% %     f = getframe();
% %     imwrite(f.cdata, fullfile('shows', [sprintf('%04d_%04d', clipIndex, i) '.png']));
% end
% % imshow(vaReadImage(clips{end}));
% % f = getframe();
% % imwrite(f.cdata, fullfile('shows', [sprintf('%04d_%04d', clipIndex, numel(clips)) '.png']));