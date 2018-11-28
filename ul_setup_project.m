function ul_setup_project()
    % change the libpath for mexopencv 
    lib_path = 'D:/Libraries/';

    % set mexopencv for ssw extraction
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
    addpath(fullfile(root, 'UL-Tracker/layers'));
end