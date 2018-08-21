function ok = ulMakeDir(path)
	ok = false;
	if exist(path, 'dir') == 0
		mkdir(path);
		ok = true;
	end
end