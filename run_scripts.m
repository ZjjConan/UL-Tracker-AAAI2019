etime = [1];

for i = 1:numel(etime)
    script_train_dcfnet(etime(i));
end

% script_extract_edgebox;
% script_remove_dead_boxes;