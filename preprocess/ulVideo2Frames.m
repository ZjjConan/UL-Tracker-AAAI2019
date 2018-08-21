function ulVideo2Frames(videoFile, varargin)
    
    opts.saveDir = '';
    opts.jpegCompressionQuality = 90;
    [opts, ~] = vl_argparse(opts, varargin);
        
    vReader = VideoReader(videoFile);
    if ~vReader.isvalid               % check if we succeeded
        error('ulVideo2Frames: cannot open file %s', videoFile);
    end
      
    [~, filename, ~] = fileparts(videoFile);
    saveDir = fullfile(opts.saveDir, filename);
    ulMakeDir(saveDir);
    f = 0; tic;
    while(hasFrame(vReader))
        f = f + 1;
        frame = readFrame(vReader);
        saveName = fullfile(saveDir, [sprintf('%08d', f) '.jpg']);
        if exist(saveName, 'file')
            if mod(f, 100) == 0
                fprintf('%s: [%s] process %08d frames time %.2fs\n', mfilename, filename, f, toc); 
            end
            continue;
        end
        imwrite(frame, saveName, 'Quality', opts.jpegCompressionQuality);
        if mod(f, 100) == 0
            fprintf('%s: [%s] process %08d frames time %.2fs\n', mfilename, filename, f, toc); 
        end
    end  
end

