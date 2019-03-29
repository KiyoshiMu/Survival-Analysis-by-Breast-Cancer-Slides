from snas import SNAS
from tools import gen_logger
import argparse
import os

def models_val(dir_p, logger, sel_num=42):
    para = os.path.basename(dir_p)
    d_size = 512 if '512' in para else 256
    model = SNAS('..', dst=dir_p, val_sel_num=sel_num, d_size=d_size)
    logger.info(f'test {dir_p} {d_size}')
    for fn in filter(lambda x:x.endswith('h5'), os.listdir(dir_p)):
        fp = os.path.join(dir_p, fn)
        logger.info(f'now {fp}')
        model.load(fp)
        model.feedback()
    logger.info(f'done {dir_p}')
        
if __name__ == "__main__":
    logger = gen_logger('main_val')
    parse = argparse.ArgumentParser()
    parse.add_argument('i', nargs='+', help='the path of directory that saves model weight for cases')
    parse.add_argument('-s', type=int, default=42, help='the num of imgs used for validation')
    command = parse.parse_args()
    logger.info(f'Begin train on {command}')
    try:
        for dir_p in command.i:
            models_val(dir_p, logger, sel_num=command.s)
    except:
        logger.exception('something wrong')