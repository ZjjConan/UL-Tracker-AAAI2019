function [filepaths, filenames] = ulDir(dirPath, ext)

    if nargin < 2
        ext = [];
    end
    
    filepaths = dir([dirPath '/*.' ext]);
    filepaths = {filepaths.name}';
    if isempty(ext)
        filepaths(1:2) = [];
    end
    if nargout == 2
        filenames = filepaths;
        filenames = cellfun(@(x) strsplit(x, '.'), filenames, 'UniformOutput', false);
        filenames = cat(1, filenames{:});
        filenames(:, 2) = [];  
    end
    
    filepaths = fullfile(dirPath, filepaths);
end

