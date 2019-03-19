# from keras.applications.nasnet import preprocess_input
import os
import cv2
import sys
sys.path.append('..')
from tools import gen_logger
from tqdm import tqdm

def img_gen(dir_p):
    cases = os.listdir(dir_p)
    for case in tqdm(cases):
        cp = os.path.join(dir_p, case)
        for f in os.listdir(cp):
            if f[-4:] == 'tiff':
                continue
            print(fp)
            break
            # try:
            #     preprocess_input(cv2.imread(fp))
            # except TypeError:
            #     logger.info(f'{fp} error')
    else:
        print('pass')
logger = gen_logger('img_test')

if __name__ == "__main__":
    img_gen(sys.argv[1])