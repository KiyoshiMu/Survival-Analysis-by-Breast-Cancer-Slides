from keras.models import Model
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.layers import Input, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Concatenate, Dropout, Dense
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import Adagrad, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.applications.nasnet import NASNetMobile, preprocess_input
import os
import cv2
import random
import pandas as pd
import numpy as np
from tools import get_seq, gen_logger
import argparse

def model_creator():
    inputs = Input((96, 96, 3))
    base_model = NASNetMobile(include_top=False, input_shape=(96, 96, 3))#, weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(1, activation="linear", kernel_initializer='glorot_uniform', 
    kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(out)
    model = Model(inputs, out)
    model.load_weights('model.h5', by_name='NASNet')
    model.layers[1].trainable = False
    model.summary()
    return model

def negative_log_likelihood(E):
	def loss(y_true,y_pred):
		hazard_ratio = K.exp(y_pred)
		log_risk = K.log(K.cumsum(hazard_ratio))
		uncensored_likelihood = K.transpose(y_pred) - log_risk
		censored_likelihood = uncensored_likelihood * E
		num_observed_event = K.sum([float(e) for e in E]) + 1
		return K.sum(censored_likelihood) / num_observed_event * (-1)
	return loss

def case2path(x_p):
    cur_dirs = os.listdir(x_p)
    result = {}
    for dir_n in cur_dirs:
        case = dir_n[:12]
        if case in result:
            continue
        result[case] = os.path.join(x_p, dir_n)
    return pd.DataFrame.from_dict(result, orient='index', columns=['path'])

def merge_table_creator(selected_p, target_p='data/Target.xlsx'):
    target = pd.read_excel(target_p)
    case_path_df = case2path(selected_p)
    merge_table = case_path_df.merge(target, left_index=True, right_on='sample')
    print(len(merge_table)/len(target))
    merge_table.reset_index(drop=True, inplace=True)
    return merge_table

def train_table_creator(merge_table, train_ratio=0.8):
    idx = merge_table.index.tolist()
    random.shuffle(idx)

    train_size = round(len(idx) * train_ratio)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    train_table = merge_table.iloc[train_idx]
    train_table.sort_values('duration', inplace=True)
    train_table.reset_index(drop=True, inplace=True)

    test_tabel = merge_table.iloc[test_idx]
    test_tabel.sort_values('duration', inplace=True)
    test_tabel.reset_index(drop=True, inplace=True)
    return train_table, test_tabel

def read_dir(dir_p, aug=True):
    pool = os.listdir(dir_p)
    sel = random.choice(pool)
    x = os.path.join(dir_p, sel)
    return cv2.imread(x)
    
def chunk(df_sort, batch_size, epochs=10):
    population = list(range(len(df_sort)))
    for _ in range(epochs):
        chunk_idx = random.choices(population, k=batch_size)
        chunk_idx.sort()
        yield df_sort.iloc[chunk_idx]

def data_gen(merge_table, batch_size, seq=None):
    for chunk_df in chunk(merge_table, batch_size):
        X, T, E = [], [], []
        for item in chunk_df.iterrows():
            path = item[1][0]
            dur = item[1][2]
            obs = item[1][3]
            X.append(read_dir(path))
            T.append(dur)
            E.append(obs)
        if seq:
            X = seq.augment_images(X)
        X = [preprocess_input(x) for x in X]
        yield np.array(X), np.array(T), E

def train(selected_p, dst='..'):
    merge_table = merge_table_creator(selected_p)
    train_table, test_tabel = train_table_creator(merge_table)
    ada = Adagrad(lr=1e-3, decay=0.1)
    seq = get_seq()
    model = model_creator()
    cheak_list = [EarlyStopping(monitor='loss', patience=10),
    ModelCheckpoint(filepath=os.path.join(dst, 'toy.h5')
    , monitor='loss', save_best_only=True),
    TensorBoard(log_dir=os.path.join(dst, 'toy_log'), 
    histogram_freq=0)]

    for (X, Y, E), (X_val, Y_val, _) in zip(data_gen(train_table, 128, seq=seq), 
    data_gen(test_tabel, 128)):
        model.compile(loss=negative_log_likelihood(E), optimizer=ada)
        model.fit(
            X, Y,
            batch_size=len(E),
            epochs=10,
            verbose=True,
            callbacks=cheak_list,
            shuffle=False,
            validation_data=(X_val, Y_val))

logger = gen_logger('train')
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('i')
    parse.add_argument('-o', default='..')
    command = parse.parse_args()
    try:
        train(command.i, dst=command.o)
    except:
        logger.exception('something wrong')