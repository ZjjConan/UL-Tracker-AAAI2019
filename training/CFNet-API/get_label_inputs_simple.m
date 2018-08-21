function inputs = get_label_inputs_simple(labels, obj_sz_z, obj_sz_x, pos_eltwise, neg_eltwise, wp_eltwise, wn_eltwise)
    pos = (labels > 0);
    neg = (labels < 0);

    resp_sz = size(pos_eltwise);
    eltwise_labels = zeros([resp_sz, 1, numel(labels)], 'single');
    eltwise_labels(:,:,:,pos) = repmat(pos_eltwise, [1 1 1 sum(pos)]);
    eltwise_labels(:,:,:,neg) = repmat(neg_eltwise, [1 1 1 sum(neg)]);
    eltwise_weights = zeros([resp_sz, 1, numel(labels)], 'single');
    eltwise_weights(:,:,:,pos) = repmat(wp_eltwise, [1 1 1 sum(pos)]);
    eltwise_weights(:,:,:,neg) = repmat(wn_eltwise, [1 1 1 sum(neg)]);
    inputs = {'label', labels, ...
              'eltwise_label', eltwise_labels, ...
              'eltwise_weight', eltwise_weights, ...
              'exemplar_size', obj_sz_z, ...
              'instance_size', obj_sz_x};
end