from keras.models import Sequential
import keras.backend as K
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.regularizers import l2
from keras.optimizers import Adagrad
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, sys

#Loss Function
def negative_log_likelihood(E):
	def loss(y_true,y_pred):
		hazard_ratio = K.exp(y_pred)
		log_risk = K.log(K.cumsum(hazard_ratio))
		uncensored_likelihood = K.transpose(y_pred) - log_risk
		censored_likelihood = uncensored_likelihood * E
		
		return K.sum(censored_likelihood) / K.sum(E) * (-1)
	return loss

def gen_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform',
    input_shape=input_shape))
    model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(1, activation="linear", kernel_initializer='glorot_uniform', 
    kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))

    return model

def eval(model, x, y, e):
    hr_pred=model.predict(x)
    hr_pred=np.exp(hr_pred)
    ci=concordance_index(y,-hr_pred,e)
    return ci

def gen_data(fp):
    df = pd.read_excel(fp)
    y = (df.loc[:, ["duration","observed"]]).values
    x = df.loc[:,'epi_area':'hist_type']
    x = (pd.get_dummies(x, drop_first=True)).values
    return x, y

def run_model(fp, dst):
    x, t_e = gen_data(fp)
    x_train, x_test, t_e_train, t_e_test = train_test_split(x, t_e, test_size=0.2, random_state=42)
    print(t_e_train[:10])
    y_train, e_train = t_e_train[:, 0], t_e_train[:, 1]
    y_test, e_test = t_e_test[:, 0], t_e_test[:, 1]

    sort_idx = np.argsort(y_train)[::-1] #!
    x_train = x_train[sort_idx]
    y_train = y_train[sort_idx]
    e_train = e_train[sort_idx]

    x_t_shape = np.shape(x_train)
    print('{} training images have prepared, shape is {}\
    and {}'.format(len(x_train), x_t_shape, np.shape(y_train)))
    print('{} test images have prepared, shape is {}\
    and {}'.format(len(x_test), np.shape(x_test), np.shape(y_test)))

    model = gen_model(x_t_shape[1:])
    ada = Adagrad(lr=1e-3, decay=0.1)
    model.compile(loss=negative_log_likelihood(e_train), optimizer=ada)

    cheak_list = [EarlyStopping(monitor='loss', patience=10),
                ModelCheckpoint(filepath=os.path.join(dst, 'toy.h5')
                , monitor='loss', save_best_only=True),
                TensorBoard(log_dir=os.path.join(dst, 'toy_log'), 
                histogram_freq=0)]

    model.fit(
        x_train, y_train,
        batch_size=len(e_train),
        epochs=10000,
        callbacks=cheak_list,
        shuffle=False)
    
    ci = eval(model, x_train, y_train, e_train)
    ci_val = eval(model, x_test, y_test, e_test)

    with open(os.path.join(dst, 'toy_outcome.txt'), 'w+') as out:
        line = 'Concordance Index for training dataset:{},\
                Concordance Index for test dataset:{}'.format(ci, ci_val)
        out.write(line)

if __name__ == "__main__":
    fp = sys.argv[1]
    dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    run_model(fp, dst)