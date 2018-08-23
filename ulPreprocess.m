clc; clear all; close all;
% convert video to frames
% opts.dataDir = 'D:/Dataset/Video/movie-dataset/movie_org/';
% opts.saveDir = 'D:/Dataset/Video/movie-dataset/images/';
opts.boxDir = 'D:\Dataset\Video\movie-dataset\boxes-clean\';
% videoNames = {'DK (2008).mkv', 'GF (1972).mkv', 'GF-II (1974).mkv', 'LR-III (2003).mkv', 'SR (1994).mkv'};
videoNames = {'DK (2008)', 'GF (1972)', 'GF-II (1974)', 'LR-III (2003)', 'SR (1994)'};
% for i = 1:5
%     boxDir = fullfile(opts.boxDir, videoNames{i});
%     boxFiles = dir([boxDir '/*.mat']);
%     boxFiles = fullfile(boxDir, {boxFiles.name});
%     for j = 1:numel(boxFiles)
%         load(boxFiles{j});
%     end
% end


% for i = 1:5
%     ulVideo2Frames(fullfile(opts.dataDir, videoNames{i}), opts);
% end


% opts.dataDir = 'D:/Dataset/Video/movie-dataset/images/';
% opts.saveDir = 'D:/Dataset/Video/movie-dataset/boxes/';
% % opts.debug = false;
% 
% ssw options
% opts.maxImageSize = 500;
% for i = 4
%     filename = strsplit(videoNames{i}, '.');
%     ulExtractSSWBoxes(fullfile(opts.dataDir, filename{1}), opts);
% end

opts.imgDir = 'D:\Dataset\Video\movie-dataset\images\';
opts.boxDir = 'D:\Dataset\Video\movie-dataset\boxes\';
opts.saveDir = 'D:\Dataset\Video\movie-dataset\boxes-clean\';

% params
params.removeWithMeanIntensity = true;
params.intensityRange = [30 220];
params.removeWithRatio = true;
params.boxRatio = [0.1 0.7];
params.removeWithBorder = false;
% 
for i = 5
    [~, vname, ~] = fileparts(videoNames{i});
    
    ulRemoveBoxes('movieName', vname, ...
                  'imgDir', fullfile(opts.imgDir, vname), ...
                  'boxDir', fullfile(opts.boxDir, vname), ...
                  'saveDir', fullfile(opts.saveDir, vname), ...
                  params);
end
% 
% 
% % segment videos
% opts.imgDir = 'D:\Dataset\Video\movie-dataset\images';
% opts.boxDir = 'D:\Dataset\Video\movie-dataset\boxes_clean\';
% opts.saveDir = 'data/clips';
% 
% params.useCorr = true;
% params.batchSize = 100;
% params.corrMinThre = 0.3;
% params.resizeRatio = 0.1;
% params.debug = false;
% videoNames = {'SR (1994)'};
% for i = 1
%     [~, vname, ~] = fileparts(videoNames{i});
%     params.imgDir = fullfile(opts.imgDir, vname);
%     params.boxDir = fullfile(opts.boxDir, vname);
%     params.saveDir = fullfile(opts.saveDir, [vname '_clips.mat']);
%     groupVideoFrames(params);
% end

