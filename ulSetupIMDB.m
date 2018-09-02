clc; clear all; close all;

videoNames = {'SR (1994)'};

% setup or load imdb
imdbOpts.imgDir = 'C:\SR (1994) - Resized';
imdbOpts.boxDir = 'D:\Dataset\Video\movie-dataset\boxes\SR (1994)';
imdbOpts.clipDir = ['data/clips/' videoNames{1} '_clips.mat'];
imdbOpts.imdbPath = ['data/imdb/' videoNames{1} '_nobbrm_imdb.mat'];
imdbOpts.isSegment = true;
imdbOpts.minFramesPerSegment = 10;

imdb = setupMovieImdb(imdbOpts);
