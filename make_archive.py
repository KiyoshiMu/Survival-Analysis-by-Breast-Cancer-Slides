import os
import shutil
import sys
from tools import load_pickle, gen_logger

logger = gen_logger(f'{__name__}.log')

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

if __name__ == "__main__":
    make_archive(sys.argv[1])