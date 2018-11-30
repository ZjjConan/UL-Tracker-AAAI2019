% clc; clear all; close all;
% convert video to frames
% opts.dataDir = 'D:/Dataset/Video/movie-dataset/movie_org/';
% opts.saveDir = 'D:/Dataset/Video/movie-dataset/images/';
% videoName = 'SR (1994).mkv';
% ul_video2frames(fullfile(opts.dataDir, videoName), opts);

clear all;

% extract region proposals using SSW
% opencv and mexopencv needed
% opts.dataDir = 'D:/Dataset/Video/movie-dataset/images/';
% opts.saveDir = 'D:/Dataset/Video/movie-dataset/boxes/';
% opts.debug = false;
% opts.useFastStrategy = true;
% opts.maxImageSize = 500;
% opts.useThreads = 4;
% videoName = 'SR (1994)';
% ul_extract_proposals(fullfile(opts.dataDir, videoName), opts);

clear all;

% get short video clips
opts.imgDir = 'C:/SR (1994) - Resized/';
opts.saveDir = 'data/clips/SR (1994).mat';
opts.useCorr = true;
opts.batchSize = 100;
opts.corrMinThre = 0.3;
opts.resizeRatio = 0.2;
opts.debug = false;
ul_group_frames(opts);


