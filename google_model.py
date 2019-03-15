import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Concatenate, Dropout, Dense
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.applications.nasnet import NASNetMobile, preprocess_input
from keras.optimizers import Adam
from tools import get_files, save_pickle, load_pickle, gen_logger
import sys
import pickle

logger = gen_logger('outcome++')

def predict(X):
    return ((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()

def chunk(case_dir, n, done=None):
    files = get_files(case_dir, suffix='tiff')
    for i in range(n, len(files), n):
        pps = files[i-n:i]
        yield pps

def get_model_classif_nasnet():
    inputs = Input((96, 96, 3))
    base_model = NASNetMobile(include_top=False, input_shape=(96, 96, 3))#, weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['acc'])
    model.summary()

    return model

def main(dir_p, dst='c:/'):
#     done = load_pickle('data/done.pkl')
    for case_name in os.listdir(dir_p):
        case_dir = os.path.join(dir_p, case_name)
        with open(os.path.join(dst, 'outcome_++.pkl'), 'ab') as temp_pkl:
            for batch in chunk(case_dir, 32):
                try:
                    X = [preprocess_input(cv2.imread(x)) for x in batch]
                    X = np.array(X)
                    preds_batch = predict(X)
                    record = {key:value for key, value in zip(batch, preds_batch)}
                    pickle.dump(record, temp_pkl, pickle.HIGHEST_PROTOCOL)
                except:
                    logger.exception(f'{case_name} encounter mistakes')
        logger.info(f'{case_name} is completed')

if __name__ == "__main__":
    model = get_model_classif_nasnet()
    h5_path = "model.h5"
    model.load_weights(h5_path)
    main(sys.argv[1], sys.argv[2])