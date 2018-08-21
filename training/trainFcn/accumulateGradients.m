% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
%     numGpus = numel(params.gpu) ;
%     otherGpus = setdiff(1:numGpus, labindex) ;

    for p=1:numel(net.params)

      if ~isempty(parserv)
        parDer = parserv.pullWithIndex(p) ;
      else
        parDer = net.params(p).der ;
      end

      switch net.params(p).trainMethod

        case 'average' % mainly for batch normalization
          thisLR = net.params(p).learningRate ;
          net.params(p).value = vl_taccum(...
              1 - thisLR, net.params(p).value, ...
              (thisLR/batchSize/net.params(p).fanout),  parDer) ;

        case 'gradient'
          thisDecay = params.weightDecay * net.params(p).weightDecay ;
          thisLR = params.learningRate * net.params(p).learningRate ;

          if thisLR>0 || thisDecay>0
            % Normalize gradient and incorporate weight decay.
            parDer = vl_taccum(1/batchSize, parDer, ...
                               thisDecay, net.params(p).value) ;

            if isempty(params.solver)
              % Default solver is the optimised SGD.
              % Update momentum.
              state.solverState{p} = vl_taccum(...
                params.momentum, state.solverState{p}, ...
                -1, parDer) ;

              % Nesterov update (aka one step ahead).
              if params.nesterovUpdate
                delta = params.momentum * state.solverState{p} - parDer ;
              else
                delta = state.solverState{p} ;
              end

              % Update parameters.
              net.params(p).value = vl_taccum(...
                1,  net.params(p).value, thisLR, delta) ;

            else
              % call solver function to update weights
              [net.params(p).value, state.solverState{p}] = ...
                params.solver(net.params(p).value, state.solverState{p}, ...
                parDer, params.solverOpts, thisLR) ;
            end
          end
        otherwise
          error('Unknown training method ''%s'' for parameter ''%s''.', ...
            net.params(p).trainMethod, ...
            net.params(p).name) ;
      end
    end
end