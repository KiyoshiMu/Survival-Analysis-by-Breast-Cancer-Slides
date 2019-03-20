from keras.models import Model, Sequential
from keras.layers import Input, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Concatenate, Dropout, Dense, Conv2D, BatchNormalization, MaxPooling2D
from keras.applications.nasnet import NASNetMobile
from keras.regularizers import l2
from keras import backend as K

def negative_log_likelihood(E):
	def loss(y_true,y_pred):
		hazard_ratio = K.exp(y_pred)
		log_risk = K.log(K.cumsum(hazard_ratio))
		uncensored_likelihood = K.transpose(y_pred) - log_risk
		censored_likelihood = uncensored_likelihood * E
		num_observed_event = K.sum([float(e) for e in E]) + 1
		return K.sum(censored_likelihood) / num_observed_event * (-1)
	return loss

def model_pns():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(96, 96, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear", kernel_initializer='glorot_uniform', 
    kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01)))
    model.summary()
    return model

def model_nas():
    inputs = Input((96, 96, 3))
    base_model = NASNetMobile(include_top=False, input_shape=(96, 96, 3))#, weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(512, kernel_initializer='glorot_uniform', 
    kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(out)
    out = Dense(1, activation="linear", kernel_initializer='glorot_uniform', 
    kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(out)
    model = Model(inputs, out)
    model.load_weights('model.h5', by_name='NASNet')
    model.layers[1].trainable = False
    model.summary()
    return model

def model_nas_clf():
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
    model.load_weights('model.h5')
    model.summary()
    return model