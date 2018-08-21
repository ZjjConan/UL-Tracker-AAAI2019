clc; clear all; close all;

videoNames = {'SR (1994)'};

% setup or load imdb
imdbOpts.imgDir = 'C:\SR (1994) - Resized';
imdbOpts.boxDir = fullfile('D:/Dataset/Video/movie-dataset/boxes-clean/', videoNames{1});
imdbOpts.clipDir = ['data/clips/' videoNames{1} '_clips.mat'];
imdbOpts.imdbPath = ['data/imdb/' videoNames{1} '_imdb.mat'];
imdbOpts.isSegment = true;
imdbOpts.minFramesPerSegment = 10;

imdb = setupMovieImdb(imdbOpts);
