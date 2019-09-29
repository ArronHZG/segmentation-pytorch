import os


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
