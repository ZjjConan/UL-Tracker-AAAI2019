clc; clear all; close all;

videoNames = {'LR-3 (2003)'};

% setup or load imdb
imdbOpts.imgDir = 'D:\Dataset\Video\movie-dataset\images\LR-3 (2003) - Resized/';
imdbOpts.boxDir = 'D:\Dataset\Video\movie-dataset\boxes\LR-3 (2003  )';
imdbOpts.clipDir = ['data/clips/' videoNames{1} '_clips.mat'];
imdbOpts.imdbPath = ['data/imdb/' videoNames{1} '_imdb.mat'];
imdbOpts.isSegment = true;
imdbOpts.minFramesPerSegment = 10;

imdb = setupMovieImdb(imdbOpts);
