def get_subdirs(p):
    ex_folders = []
    for it in os.scandir(p):
        if it.is_dir():
            # it.name or it.path
            ex_folders.append(it.path)

    return ex_folders
