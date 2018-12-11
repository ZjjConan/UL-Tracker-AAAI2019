% create imdb for network training
clc; clear all; close all;
opts.imgDir = 'C:/SR (1994)/';
opts.boxDir = 'D:\Dataset\Video\movie-dataset\boxes/SR (1994) - proposals/';

opts.clipDir = 'data/clips/SR (1994).mat';
opts.imdbPath = 'data/imdb/SR (1994)_imdb.mat';
opts.isSegment = true;
opts.minFramesPerSegment = 10;
% we rescale all images with this ratio, so all proposals should multiply
% this ratio.
opts.scaleRatio = 0.5;

imdb = create_imdb(opts);
