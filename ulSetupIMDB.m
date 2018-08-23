clc; clear all; close all;

videoNames = {'SR (1994)'};

% setup or load imdb
imdbOpts.imgDir = '/home/zjjconan/UL-Tracker/SR (1994) - Resized';
imdbOpts.boxDir = '/home/zjjconan/UL-Tracker/SR (1994) - Box';
imdbOpts.clipDir = ['data/clips/' videoNames{1} '_clips.mat'];
imdbOpts.imdbPath = ['data/imdb/' videoNames{1} '_imdb.mat'];
imdbOpts.isSegment = true;
imdbOpts.minFramesPerSegment = 10;

imdb = setupMovieImdb(imdbOpts);
