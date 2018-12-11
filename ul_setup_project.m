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
    addpath(fullfile(root, 'UL-Tracker-AAAI2019/preprocess'));
    addpath(fullfile(root, 'UL-Tracker-AAAI2019/utils'));
    addpath(genpath(fullfile(root, 'UL-Tracker-AAAI2019/training')));
    addpath(fullfile(root, 'UL-Tracker-AAAI2019/layers'));
end