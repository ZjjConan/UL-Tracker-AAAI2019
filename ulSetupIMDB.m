clc; clear all; close all;

videoNames = {'SR (1994)'};

% setup or load imdb
if ispc
    imdbOpts.imgDir = 'C:\SR (1994) - Resized/';
    imdbOpts.boxDir = 'D:\Dataset\Video\movie-dataset\boxes\SR (1994)';
else
    
end
imdbOpts.clipDir = ['data/clips/' videoNames{1} '_clips_0.9.mat'];
imdbOpts.imdbPath = ['data/imdb/' videoNames{1} '_imdb_0.9.mat'];
imdbOpts.isSegment = true;
imdbOpts.minFramesPerSegment = 10;

imdb = setupMovieImdb(imdbOpts);
