from keras.models import Sequential
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.layers import MaxPooling2D, SeparableConv2D, Flatten, Dense, Dropout, BatchNormalization
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import Adagrad, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
import os ,sys, cv2, random
import matplotlib.pyplot as plt

#Loss Function
def negative_log_likelihood(E):
	def loss(y_true,y_pred):
		hazard_ratio = K.exp(y_pred)
		log_risk = K.log(K.cumsum(hazard_ratio))
		uncensored_likelihood = K.transpose(y_pred) - log_risk
		censored_likelihood = uncensored_likelihood * E
		num_observed_event = K.sum([int(e) for e in E]) or 1
		return K.sum(censored_likelihood) / num_observed_event * (-1)
	return loss
    
"""SCNN model, a much small structure of PNS"""
# complex
def gen_model():
    model = Sequential()
    model.add(SeparableConv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(SeparableConv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(SeparableConv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="linear", kernel_initializer='glorot_uniform', 
    kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))
    return model

def read_dir(dir_path, time):
    data = []
    pool = random.sample(dir_path, time)
    for imagePath in os.listdir(pool):
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (256, 256))
        image = img_to_array(image)
        data.append(image)
    sample = [int(os.path.basename(f).split('.')[0]) for f in pool]
    return np.array(data, dtype="float"), sample

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

def gen_data(x_p, y_dataset, amount):
    x, samples = read_dir(x_p, amount)
    y = y_dataset[samples]
    E = y['event'].values
    T = y['duration'].values
    #Sorting for NNL!
    sort_idx = np.argsort(T)[::-1] #!
    x = x[sort_idx]
    T = T[sort_idx]
    E = E[sort_idx]

    return x, T, E

def eval(model, x, y, e):
    hr_pred=model.predict(x)
    hr_pred=np.exp(hr_pred)
    ci=concordance_index(y,-hr_pred,e)
    return ci

def run_model(x_train_p, y_train_p, x_val_p, y_val_p, dst, batch_size, epochs=300):
    model = gen_model()
    print(model.summary())
    y_train_dataset = pd.read_csv(y_train_p, delimiter=r'\s+', index_col=0)
    y_val_dataset = pd.read_csv(y_val_p, delimiter=r'\s+', index_col=0)
    x_train_p = [os.path.join(x_train_p, f) for f in os.listdir(x_train_p)]
    x_val_p = [os.path.join(x_val_p, f) for f in os.listdir(x_val_p)]
    ada = Adagrad(lr=1e-3, decay=0.1)
    # rmsprop=RMSprop(lr=1e-5, rho=0.9, epsilon=1e-8)
    cheak_list = [EarlyStopping(monitor='loss', patience=10),
                ModelCheckpoint(filepath=os.path.join(dst, 'modelcc.h5')
                , monitor='loss', save_best_only=True),
                TensorBoard(log_dir=os.path.join(dst, 'cell_log'), 
                histogram_freq=0)]
    aug = data_flow()

    for _ in range(epochs):
        model.compile(loss=negative_log_likelihood(e_train), optimizer=ada)
        x_train, y_train, e_train = gen_data(x_train_p, y_train_dataset, batch_size)
        x_val, y_val, e_val = gen_data(x_val_p, y_val_dataset, amount=100)
    # print('{} training images have prepared, shape is {}\
    # and {}'.format(len(x_train), np.shape(x_train), np.shape(y_train)))
    # print('{} validation images have prepared, shape is {}\
    # and {}'.format(len(x_val), np.shape(x_val), np.shape(y_val)))
        model.fit_generator(
        aug.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=1,
        epochs=1,
        callbacks=cheak_list,
        shuffle=False)
    
    ci = eval(model, x_train, y_train, e_train)
    ci_val = eval(model, x_val, y_val, e_val)
    with open(os.path.join(dst, 'outcome.txt'), 'w+') as out:
        line = 'Concordance Index for training dataset:{},\
                Concordance Index for validation dataset:{}'.format(ci, ci_val)
        out.write(line)

    # plot_process(history, dst)

def main():
    x_train_p = sys.argv[1]
    y_train_p = sys.argv[2]
    x_val_p = sys.argv[3]
    y_val_p = sys.argv[4]
    dst = sys.argv[5]
    batch_size = int(sys.argv[6])
    os.makedirs(dst, exist_ok=True)
    run_model(x_train_p, y_train_p, x_val_p, y_val_p, dst, batch_size)

if __name__ == "__main__":
    
    main()