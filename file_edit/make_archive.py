import os
import shutil
import argparse
from tools import gen_logger
from tqdm import tqdm

logger = gen_logger('make_archive')

def make_archive(cases_p, archive_p):
    os.makedirs(archive_p, exist_ok=True)
    for case in tqdm(os.listdir(cases_p)):
        try:
            shutil.make_archive(os.path.join(archive_p, case.split('.')[0]), 
            'zip', os.path.join(cases_p, case))
            logger.info(f'{case} is completed')
        except:
            logger.exception(f'{case} encounts some errors')

def fn_match(slides_p):
    slides = os.listdir(slides_p)
    return {fn.split('.')[0]:fn for fn in slides}

def copy_special_slide(slides_p, archive_p):
    archive = os.listdir(archive_p)
    special_p = f'{os.path.pardir(archive_p)}/slide_special'
    os.makedirs(special_p, exist_ok=True)
    fn_match_dict = fn_match(slides_p)
    for fn in archive:
        if os.path.getsize(os.path.join(archive_p, fn)) < 1000:
            case = os.path.splitext(fn)[0]
            src = os.path.join(slides_p, fn_match_dict[case])
            shutil.copy(src, special_p)
            logger.info(f'{case} slide is moved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="this strip is used to archive small areas to \
make copy or move faster")
    parser.add_argument('i', help='the directory path')
    parser.add_argument('-o', default='../archive')
    parser.add_argument('-s','--slide', type=str, 
    help='after making archive, copy the abnormal .svs slide whose archives are less than \
1K byte out to the same level of dst, with name slide_special')
    command = parser.parse_args()
    make_archive(command.i, command.o)
    if command.m:
        try:
            copy_special_slide(command.s, command.o)
        except:
            logger.exception('the path of slides may not correct')