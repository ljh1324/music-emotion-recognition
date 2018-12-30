from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD


def new_jh_net(input_shape, weights_path=None):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation="relu", name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(8, (3, 3), activation="relu", name='conv1_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(8, (3, 3), activation="relu", name='conv1_3'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling1'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(8, (3, 3), activation="relu", name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(8, (3, 3), activation="relu", name='conv2_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(8, (3, 3), activation="relu", name='conv2_3'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling2'))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu', name='dense1'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', name='dense2'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax', name='dense3'))

    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = Adam(lr=0.000-1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    # adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def load_model(model_name, input_shape, weights_path=None):
    if model_name == 'new_jh_net':
        return new_jh_net(input_shape=input_shape, weights_path=weights_path)
