% create imdb for network training
clc; clear all; close all;
opts.imgDir = 'C:/SR (1994) - Resized/';
opts.boxDir = 'D:\Dataset\Video\movie-dataset\boxes/SR (1994) - proposals/';

opts.clipDir = 'data/clips/SR (1994).mat';
opts.imdbPath = 'data/imdb/SR (1994)_imdb.mat';
opts.isSegment = true;
opts.minFramesPerSegment = 10;

imdb = create_imdb(opts);
