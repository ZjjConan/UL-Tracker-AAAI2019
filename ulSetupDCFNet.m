function ulSetupDCFNet()
    if ispc
        addpath('F:/Research/tracker_zoo/DCFNet');
        addpath('F:/Research/tracker_zoo/DCFNet/training');
    else
        addpath('/media/zjjconan/Experiments/tracker_zoo/DCFNet/');
        addpath('/media/zjjconan/Experiments/tracker_zoo/DCFNet/training');
    end
    fftw('planner', 'patient');
end