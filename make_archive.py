import os
import shutil
import argparse
from tools import load_pickle, gen_logger

logger = gen_logger('make_archive.log')
def make_archive(cases_p, done='data/done.pkl'):
    done_case = load_pickle(done)
    archive_p = '../archive'
    os.makedirs(archive_p, exist_ok=True)
    for case in os.listdir(cases_p):
        if case in done_case:
            continue
        try:
            shutil.make_archive(os.path.join(archive_p, case.split('.')[0]), 
            'zip', os.path.join(cases_p, case))
            logger.info(f'{case} is completed')
        except:
            logger.exception(f'{case} encounts some errors')

def fn_match(slides_p):
    slides = os.listdir(slides_p)
    return {fn.split('.')[0]:fn for fn in slides}

def move_slide(slides_p):
    archive_p = '../archive'
    archive = os.listdir(archive_p)
    special_p = '../special'
    os.makedirs(special_p, exist_ok=True)
    fn_match_dict = fn_match(slides_p)
    for case in archive:
        if os.path.getsize(os.path.join(archive_p, case)) < 1000:
            src = os.path.join(slides_p, fn_match_dict[os.path.splitext(case)[0]])
            shutil.copy(src, special_p)
            logger.info(f'{case} slide is moved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('func',nargs=1, type=str, default='a')
    parser.add_argument('-i', '--input', nargs=1)
    # parser.add_argument('-o','--out', nargs=1)
    command = parser.parse_args()
    if command.func == 'a':
        make_archive(command.i)
    else:
        move_slide(command.i)