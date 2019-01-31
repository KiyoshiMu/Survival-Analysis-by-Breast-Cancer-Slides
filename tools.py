import logging
import os

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

def gen_logger(name='dividing.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_handler = logging.FileHandler(f'../{name}')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger