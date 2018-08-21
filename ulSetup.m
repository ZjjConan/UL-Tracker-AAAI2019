function ulSetup()
    clc; clear all; close all
    % lib root path
    if ispc
        lib_path = 'D:/Libraries/';
    elseif isunix
        lib_path = '/media/zjjconan/Experiments/Libraries/'; 
    end

    % set mexopencv
    addpath(fullfile(lib_path, 'mexopencv'));
    addpath(fullfile(lib_path, 'mexopencv/opencv_contrib'));
    
    % setup matconvnet
    matconvnet_path = fullfile(lib_path, 'matconvnet');
    run([matconvnet_path '/matlab/vl_setupnn']);
    
    
    % setup current path
    root = fileparts(fileparts(mfilename('fullpath'))) ;
    addpath(fullfile(root, 'UL-Tracker/preprocess'));
    addpath(fullfile(root, 'UL-Tracker/utils'));
    addpath(genpath(fullfile(root, 'UL-Tracker/training')));
end