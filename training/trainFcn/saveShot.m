function saveShot(net, state, stats, modelPath)

    state.stats.train = stats ;
    for i = 1:numel(state.solverState)
        s = state.solverState{i} ;
        if isnumeric(s)
            state.solverState{i} = gather(s) ;
        elseif isstruct(s)
            state.solverState{i} = structfun(@gather, s, 'UniformOutput', false) ;
        end
    end
        
    if strcmpi(net.device, 'gpu')
        useGpu = true;
    end
    
    net.reset() ;
    net.move('cpu') ;
        
    saveState(modelPath, net, state) ;
        
    stats.train = state.stats.train;
    saveStats(modelPath, stats);
                
    if useGpu
        net.move('gpu');
        for i = 1:numel(state.solverState)
            s = state.solverState{i} ;
            if isnumeric(s)
                state.solverState{i} = gpuArray(s) ;
            elseif isstruct(s)
                state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
            end
        end
    end
end

