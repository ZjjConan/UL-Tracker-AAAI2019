function ok = ul_make_dir(path)
	ok = false;
	if exist(path, 'dir') == 0
		mkdir(path);
		ok = true;
	end
end