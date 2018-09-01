clc; clear all; close all;

videoNames = {'DK (2008)'};

% setup or load imdb
imdbOpts.imgDir = '/home/zjjconan/UL-Tracker/DK (2008) - Resized';
imdbOpts.boxDir = '/home/zjjconan/UL-Tracker/DK (2008) - Box';
imdbOpts.clipDir = ['data/clips/' videoNames{1} '_clips.mat'];
imdbOpts.imdbPath = ['data/imdb/' videoNames{1} '_imdb.mat'];
imdbOpts.isSegment = true;
imdbOpts.minFramesPerSegment = 10;

imdb = setupMovieImdb(imdbOpts);
