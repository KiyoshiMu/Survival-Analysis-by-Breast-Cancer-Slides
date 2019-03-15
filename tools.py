import logging
import os
import pickle
import sys

def save_pickle(data, dst, name='record'):
    with open(os.path.join(dst, f'{name}.pkl'), 'ab') as record:
        pickle.dump(data, record, pickle.HIGHEST_PROTOCOL)

def load_pickle(pkl_path):
    with open(pkl_path, 'rb') as temp:
        container = pickle.load(temp)
    return container

def get_files(path: str, suffix='svs') -> list:
    """From a dir read the .svs files, then we can divide the large slide into small ones
    path: the path of all .svs files."""
    # for f in os.listdir(path):
    #     if f[-4:] == '.svs':
    #         yield os.path.join(path, f)
    result = [os.path.join(path, i) for i in os.listdir(path) if i.rsplit('.', 1)[-1]==suffix]
    return result

def get_name(slide_path):
    case_name = os.path.splitext(os.path.basename(slide_path))[0]
    return case_name

def gen_logger(name='dividing'):
    name=f'{name}.log'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(f'../{name}')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger