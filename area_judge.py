import numpy as np
import pandas as pd
import os
import cv2
from models import model_nas_clf
from keras.losses import binary_crossentropy
from keras.applications.nasnet import preprocess_input
from keras.optimizers import Adam
from tools import get_files, save_pickle, load_pickle, gen_logger
import argparse
import pickle

def predict(model, X):
    return ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()

def chunk(case_dir, n, done=None):
    files = get_files(case_dir, suffix='tiff')
    for i in range(n, len(files), n):
        pps = files[i-n:i]
        yield pps

def judge_area(dir_p, dst='c:/', pkl_name='outcome', logger=None):
#     done = load_pickle('data/done.pkl')
    if logger is None:
        logger = gen_logger(f'{pkl_name}')

    model = model_nas_clf()
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['acc'])

    for case_name in os.listdir(dir_p):
        case_dir = os.path.join(dir_p, case_name)
        if not os.path.isdir(case_dir):
            continue
        with open(os.path.join(dst, f'{pkl_name}.pkl'), 'ab') as temp_pkl:
            for batch in chunk(case_dir, 32):
                try:
                    X = [preprocess_input(cv2.imread(x)) for x in batch]
                    X = np.array(X)
                    preds_batch = predict(model, X)
                    record = {key:value for key, value in zip(batch, preds_batch)}
                    pickle.dump(record, temp_pkl, pickle.HIGHEST_PROTOCOL)
                except:
                    logger.exception(f'{case_name} encounter mistakes')
        logger.info(f'{case_name} is completed')

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='This stript is used for valid area detection')
    parse.add_argument('i', help='the directory path of cases')
    parse.add_argument('-o', default='..', help='the path of .pkl file')
    parse.add_argument('-n', default='outcome', help='the name of .pkl file')
    command = parse.parse_args()
    dst = command.o
    os.makedirs(dst, exist_ok=True)
    judge_area(command.i, dst=dst, pkl_name=command.n)