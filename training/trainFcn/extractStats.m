% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
for i = 1:numel(sel)
  if net.layers(sel(i)).block.ignoreAverage, continue; end;
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end