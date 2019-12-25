# cython: language_level=3
import os


def make_sure_path_exists(some_dir):
    if not os.path.exists(some_dir):
        os.makedirs(some_dir)
