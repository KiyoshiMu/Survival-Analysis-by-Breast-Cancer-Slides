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

def predicting(model, X):
    return ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()

def chunk(case_dir, n, done=None):
    files = get_files(case_dir, suffix='tiff')
    for i in range(n, len(files), n):
        pps = files[i-n:i]
        yield pps

def judge_area(dir_p, dst='c:/', logger=None):
    os.makedirs(dst, exist_ok=True)
    if logger is None:
        logger = gen_logger('judge_area')

    model = model_nas_clf()
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['acc'])

    for case_name in os.listdir(dir_p):
        case_dir = os.path.join(dir_p, case_name)
        record = {}
        if not os.path.isdir(case_dir):
            continue
        
        try:
            for batch in chunk(case_dir, 32):
                X = [preprocess_input(cv2.imread(x)) for x in batch]
                X = np.array(X)
                preds_batch = predicting(model, X)
                record.update({key:value for key, value in zip(batch, preds_batch)})
            with open(os.path.join(dst, f'{case_name}.pkl'), 'wb') as temp_pkl:
                pickle.dump(record, temp_pkl, pickle.HIGHEST_PROTOCOL)
            logger.info(f'{case_name} is completed')
        except:
            logger.exception(f'{case_name} encounter mistakes')
        
if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='This stript is used for valid area detection')
    parse.add_argument('i', help='the directory path of cases')
    parse.add_argument('-o', default='..', help='the path of .pkl file')
    command = parse.parse_args()
    dst = command.o
    os.makedirs(dst, exist_ok=True)
    judge_area(command.i, dst=dst)