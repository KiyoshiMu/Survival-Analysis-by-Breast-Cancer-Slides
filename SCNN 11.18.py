from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout, BatchNormalization
from keras.engine.topology import Layer
from keras import backend as K
from keras.optimizers import Adagrad, RMSprop
import numpy as np
from keras.layers import Lambda
import wtte.weibull as weibull
import wtte.wtte as wtte
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

def run_model():
    cur_model = gen_model()
    print(cur_model.summary())
    # cur_model.fit(x, y)

if __name__ == '__main__':
    run_model()


