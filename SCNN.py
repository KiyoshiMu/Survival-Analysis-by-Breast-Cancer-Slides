from keras.models import Sequential
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout, BatchNormalization, Lambda
from keras.engine.topology import Layer
from keras import backend as K
from keras.optimizers import Adagrad, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd
import wtte.weibull as weibull
import wtte.wtte as wtte
import os
import sys
import cv2
import matplotlib.pyplot as plt

"""custom COX layer"""
class Cox(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Cox, self).__init__(**kwargs)
        # self.input = input[0] if len(input) == 1 else K.concatenate(input, axis=1)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Cox, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        self.theta = K.dot(x, self.W) # + self.b
        self.theta = K.reshape(self.theta, [K.shape(self.theta)[0]]) #recast theta as vector
        self.exp_theta = K.exp(self.theta)
        return K.dot(K.concatenate(x, axis=1), self.W)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# tte_mean_train = np.nanmean(train_y[:,0])
# mean_u = np.nanmean(train_y[:,1])
# # Initialization value for alpha-bias 
# init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
# init_alpha = init_alpha/mean_u
# print('tte_mean_train', tte_mean_train, 'init_alpha: ',init_alpha,'mean uncensored train: ',mean_u)

"""SCNN model, mock the exact structure of PNS"""
# complex
def gen_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(250, 250, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D())
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D())

    model.add(Flatten())
    # model.add(Dense(1000))
    # model.add(Dense(1000))
    model.add(Dense(256))
    model.add(Dropout(0.05))

    # model.add(Cox(1))
    model.add(Lambda(wtte.output_lambda, 
                 arguments={
                            "max_beta_value":100.0, 
                            "alpha_kernel_scalefactor":0.5
                           },
                ))

    # Use the discrete log-likelihood for Weibull survival data as our loss function
    loss = wtte.loss(kind='discrete',reduce_loss=False).loss_function

    model.compile(loss=loss, optimizer=Adagrad(lr=.001, clipvalue=0.1))

    return model

def run_model(x_train_p, y_train_p, x_val_p, y_val_p, dst, batch_size):
    model = gen_model()
    print(model.summary())
    x_train = read_dir(x_train_p)
    y_train = (pd.read_csv(y_train_p, delimiter=r'\s+', index_col=0)).values
    x_val = read_dir(x_val_p)
    y_val = (pd.read_csv(y_val_p, delimiter=r'\s+', index_col=0)).values
    print('{} training images have prepared, shape is {}\
    and {}'.format(len(x_train), np.shape(x_train), np.shape(y_train)))
    print('{} validation images have prepared, shape is {}\
    and {}'.format(len(x_val), np.shape(x_val), np.shape(y_val)))

    cheak_list = [EarlyStopping(monitor='loss', patience=10),
                ModelCheckpoint(filepath=os.path.join(dst, 'modelcc.h5')
                , monitor='loss', save_best_only=True),
                TensorBoard(log_dir=os.path.join(dst, 'cell_log'), 
                histogram_freq=0)]

    aug = data_flow()
    history = model.fit_generator(
        aug.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=300,
        callbacks=cheak_list,
        validation_data=(x_val, y_val))
    plot_process(history, dst)
    # cur_model.fit(x, y)

def read_dir(dir_path):
    data = []
    for f in os.listdir(dir_path):
        imagePath = os.path.join(dir_path, f)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (250, 250))
        image = img_to_array(image)
        data.append(image)
    return np.array(data, dtype="float")

def data_flow():
    aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)
    return aug

def plot_process(history, dst):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(dst, 'Training and validation accuracy.png'), dpi=300)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(dst, 'Training and validation loss.png'), dpi=300)

def main():
    x_train_p = sys.argv[1]
    y_train_p = sys.argv[2]
    x_val_p = sys.argv[3]
    y_val_p = sys.argv[4]
    dst = sys.argv[5]
    batch_size = sys.argv[6]
    os.makedirs(dst, exist_ok=True)
    run_model(x_train_p, y_train_p, x_val_p, y_val_p, dst, batch_size)

if __name__ == "__main__":
    
    main()