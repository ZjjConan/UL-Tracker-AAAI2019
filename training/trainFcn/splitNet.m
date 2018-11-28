function [net1, net2] = splitNet(net, splitLayer)
    if isempty(splitLayer)
        net1 = net.copy();
        net2 = net.copy();
    else
        net1 = net.copy();
        net2 = net.copy();
        removedIndex = net1.getLayerIndex(splitLayer);
        layers = {net1.layers.name};    
        net1.removeLayer(layers(removedIndex+1:end));
        net2.removeLayer(layers(1:removedIndex));
    end
end