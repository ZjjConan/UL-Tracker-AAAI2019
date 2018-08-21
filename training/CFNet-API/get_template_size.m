function [tmpl_sz, tmpl_stride] = get_template_size(net, opts)
    sizes = net.getVarSizes({'exemplar', [opts.exemplarSize*[1 1] 3 256], ...
                             'instance', [opts.instanceSize*[1 1] 3 256]});
    tmpl_sz = sizes{net.getVarIndex('br1_out')}(1:2);
    rfs = net.getVarReceptiveFields('exemplar');
    tmpl_stride = rfs(net.getVarIndex('br1_out')).stride(1);
    assert(all(rfs(net.getVarIndex('br1_out')).stride == tmpl_stride));
end
