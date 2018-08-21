function [net, derOutputs, inputs_fn] = setup_loss(net, resp_sz, resp_stride, crop_sz_z, crop_sz_x, loss_opts)
% Add layers to the network and constructs a function that returns the inputs required by the loss layer.


    switch loss_opts.type
        case 'simple'
            %% create label and weights for logistic loss
            net.addLayer('objective', ...
                         LogisticLoss(), ...
                         {'score', 'eltwise_label', 'eltwise_weight'}, 'objective');
            % adding weights to loss layer
            [pos_eltwise, neg_eltwise, pos_weight, neg_weight] = create_labels(...
                resp_sz, loss_opts.labelWeight, ...
                loss_opts.rPos/resp_stride, loss_opts.rNeg/resp_stride);

            derOutputs = {'objective', 1};
            inputs_fn = @(labels, obj_sz_z, obj_sz_x) get_label_inputs_simple(...
                labels, obj_sz_z, obj_sz_x, pos_eltwise, neg_eltwise, pos_weight, neg_weight);

        otherwise
            error('Unknown loss')
    end

    switch loss_opts.type
        case 'simple'
            net.addLayer('errdisp', centerThrErr('stride', resp_stride), ...
                         {'score','label'}, 'errdisp');
            net.addLayer('iou', IOUErrorScore('stride', resp_stride), ...
                         {'score', 'label', 'exemplar_size', 'instance_size'}, ...
                         'iou');
        otherwise
            error('Unknown loss')
    end
end