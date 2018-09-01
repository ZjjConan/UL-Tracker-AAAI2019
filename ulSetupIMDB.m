clc; clear all; close all;

videoNames = {'DK (2008)'};

% setup or load imdb
<<<<<<< HEAD
imdbOpts.imgDir = 'C:\SR (1994) - Resized';
imdbOpts.boxDir = 'D:\Dataset\Video\movie-dataset\boxes-clean\SR (1994)';
=======
imdbOpts.imgDir = '/home/zjjconan/UL-Tracker/DK (2008) - Resized';
imdbOpts.boxDir = '/home/zjjconan/UL-Tracker/DK (2008) - Box';
>>>>>>> 329c8ae1a13ad8b5c5dabaa219ec9c00da535ad7
imdbOpts.clipDir = ['data/clips/' videoNames{1} '_clips.mat'];
imdbOpts.imdbPath = ['data/imdb/' videoNames{1} '_imdb.mat'];
imdbOpts.isSegment = true;
imdbOpts.minFramesPerSegment = 10;

imdb = setupMovieImdb(imdbOpts);
