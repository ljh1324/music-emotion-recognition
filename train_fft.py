from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from music_project.extract_fft_feature import load_data
from music_project.extract_fft_feature import data_rows, data_cols
from keras.models import Model

from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint
# from keras import backend as K

import numpy as np
import os

"""
<Error>
ValueError: Negative dimension size caused by subtracting 2 from 1 for 'max_pooling2d_2/MaxPool' (op: 'MaxPool') with input shapes: [?,1,112,128].

<Solve>
Quoting an answer mentioned in github, you need to specify the dimension ordering:

Keras is a wrapper over Theano or Tensorflow libraries. Keras uses the setting variable image_dim_ordering to decide if the input layer is Theano or Tensorflow format. This setting can be specified in 2 ways -

specify 'tf' or 'th' in ~/.keras/keras.json like so -  image_dim_ordering: 'th'. Note: this is a json file.
or specify the image_dim_ordering in your model like so: model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
Appendix: image_dim_ordering in 'th' mode the channels dimension (the depth) is at index 1 (e.g. 3, 256, 256). In 'tf' mode is it at index 3 (e.g. 256, 256, 3). Quoting @naoko from comments.

* model.add(MaxPooling2D((2,2), strides=(2,2)) ==> model.add(MaxPooling2D((2,2), strides=(2,2), dim_ordering="th"))

<Reference>
https://github.com/fchollet/keras/issues/3945
https://stackoverflow.com/questions/39815518/keras-maxpooling2d-layer-gives-valueerror

"""

"""
<Error>
ImportError: `load_weights` requires h5py.

<Solve>
Have you tried directly installing h5py? http://docs.h5py.org/en/latest/build.html

Try running pip install h5py

* pip install h5py

<Reference>
https://github.com/fchollet/keras/issues/3426

"""
# K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def vgg_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    if weights_path:
        model.load_weights(weights_path)

    return model


def small_vgg_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(180, 180, 1)))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def test_model(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(180, 180, 1)))
    model.add(Conv2D(16, (3, 3), activation="relu", name='conv2d_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, (3, 3), activation="relu", name='conv2d_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', name='my_dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', name='my_dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


# 축을 하나 늘린다.
def pre_process(imgs):
    # image 개수, image row, image col 만큼 할당
    imgs = imgs[..., np.newaxis]    # 이미지에 새로운 축 추가(vgg19에 맞도록)
    return imgs


def train_and_predict():
    data_path = 'total_genie_music'

    save_train_img_file_name = "train.npy"
    save_train_result_file_name = "train_result.npy"

    save_test_img_file_name = "test.npy"
    save_test_result_file_name = "test_result.npy"

    print('-' * 30)
    print(' Loding Train Image Set')
    print('-' * 30)

    train_imgs, train_result = load_data(data_path, save_train_img_file_name, save_train_result_file_name)
    train_imgs = train_imgs.astype('float32')
    train_imgs = pre_process(train_imgs)

    """
    mean = np.mean(train_imgs)  # mean for data centering
    std = np.std(train_imgs)  # std for data normalization

    train_imgs -= mean  # nomalization
    train_imgs /= std
    """

    print(train_imgs.shape)
    print(train_result.shape)

    model = small_vgg_19()
    plot_model(model, to_file='model.png')
    model_checkpoint = ModelCheckpoint('weights_fft.h5', monitor='val_loss', save_best_only=True)

    print('-' * 30)
    print(' Fitting Model')
    print('-' * 30)

#    model.load_weights('weights.h5')
    for i in range(10000):
        model.fit(train_imgs, train_result, batch_size=4, epochs=10, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        model.save_weights('weights_fft.h5')

    print('-' * 30)
    print(' Loding Test Image Set')
    print('-' * 30)

    test_imgs, test_result = load_data(data_path, save_test_img_file_name, save_test_result_file_name)
    test_imgs = test_imgs.astype('float32')
    test_imgs = pre_process(test_imgs)

    """
    mean = np.mean(test_imgs)  # mean for data centering
    std = np.std(test_imgs)  # std for data normalization

    test_imgs -= mean  # nomalization
    test_imgs /= std
    """

    print('-' * 30)
    print(' Run Model')
    print('-' * 30)
    predict_result = model.predict(test_imgs)

    print('-' * 30)
    print(' Evaluate Accuracy')
    print('-' * 30)
    test_arg_max = np.argmax(test_result, axis=1)
    predict_arg_max = np.argmax(predict_result, axis=1)

    total_len = test_imgs.shape[0]

    test_result = np.equal(test_arg_max, predict_arg_max)

    print(' Test Set Accuracy: ' + str(np.sum(test_result) / total_len))


def save_model(file_path, num, data_set, model):
    for layer in model.layers:
        print(layer.get_output_at(0))
    save_model = Sequential()
    save_model.add(ZeroPadding2D((1, 1), input_shape=(180, 180, 1)))
    save_model.add(Conv2D(16, (3, 3), activation="relu", weights=model.get_layer("conv2d_1").get_weights(), name='conv2d_1'))

    data_path = os.path.join(file_path, 'conv2d_1')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    f_write = open(os.path.join(data_path, str(num) + '.csv'), 'w')
    result = save_model.predict(data_set)
    print(result.shape)

    save_model.add(ZeroPadding2D((1, 1)))
    save_model.add(Conv2D(16, (3, 3), activation="relu", weights=model.get_layer("conv2d_2").get_weights(), name='conv2d_2'))

    data_path = os.path.join(file_path, 'conv2d_2')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    f_write = open(os.path.join(data_path, str(num) + '.csv'), 'w')
    result = save_model.predict(data_set)
    print(result.shape)

    save_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    save_model.add(Flatten())
    save_model.add(Dense(1024, activation='relu', weights=model.get_layer('my_dense_1').get_weights(), name='my_dense_1'))
    data_path = os.path.join(file_path, 'dense_1')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    f_write = open(os.path.join(data_path, str(num) + '.csv'), 'w')
    result = save_model.predict(data_set)
    print(result.shape)

    save_model.add(Dense(1024, activation='relu', weights=model.get_layer('my_dense_2').get_weights(), name='my_dense_2'))
    data_path = os.path.join(file_path, 'dense_2')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    f_write = open(os.path.join(data_path, str(num) + '.csv'), 'w')
    result = save_model.predict(data_set)
    print(result.shape)

    """
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
    layer_outs = functor([data_set, 1.])
    print(layer_outs.shape)
    """
    """
    for layer in model.layers:
        dir_path = os.path.join(file_path, 'layer ' + str(idx))
        print(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f_write = open(os.path.join(dir_path, str(num) + '.txt'), 'w')

        save_model.add(layer)
        result = save_model.predict(data_set)
        print(result.shape)
    """


# https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
def save_intermediate_layer_output(model, data, layer_names, save_dir_name, iteration):
    for layer_name in layer_names:
        dir = os.path.join(save_dir_name, layer_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        f_write = open(os.path.join(dir, str(iteration) + '.csv'), 'w')
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(data)
        output_shape = intermediate_output.shape
        np.set_printoptions(suppress=True)
        if len(output_shape) == 2:
            f_write.write(str(output_shape[0]) + ',' + str(output_shape[1]) + '\n')
            for i in range(output_shape[0]):
                for j in range(output_shape[1]):
                    if i == output_shape[0] - 1 and j == output_shape[1] - 1:
                        print('tt')
                        f_write.write("%f" % intermediate_output[i, j])
                    else:
                        f_write.write("%f" % intermediate_output[i, j] + ',')
                f_write.write('\n')

        if len(output_shape) == 3:
            f_write.write(str(output_shape[0]) + ',' + str(output_shape[1]) + ',' + str(output_shape[2]))
            for k in range(output_shape[2]):
                for i in range(output_shape[0]):
                    for j in range(output_shape[1]):
                        if i == output_shape[0] - 1 and j == output_shape[1] - 1:
                            f_write.write("%f" % intermediate_output[i, j, k])
                        else:
                            f_write.write("%f" % intermediate_output[i, j, k] + ',')
                f_write.write('\n')


def train_and_save_result():
    data_path = 'total_genie_music'

    save_train_img_file_name = "train.npy"
    save_train_result_file_name = "train_result.npy"

    save_test_img_file_name = "test.npy"
    save_test_result_file_name = "test_result.npy"

    print('-' * 30)
    print(' Loding Train Image Set')
    print('-' * 30)

    train_imgs, train_result = load_data(data_path, save_train_img_file_name, save_train_result_file_name)
    train_imgs = train_imgs.astype('float32')
    train_imgs = pre_process(train_imgs)

    """
    mean = np.mean(train_imgs)  # mean for data centering
    std = np.std(train_imgs)  # std for data normalization

    train_imgs -= mean  # nomalization
    train_imgs /= std
    """

    print(train_imgs.shape)
    print(train_result.shape)

    model = test_model()

    print('-' * 30)
    print(' Fitting Model')
    print('-' * 30)

#    model.load_weights('weights.h5')
    for i in range(10000):
        # save_model('output_set', 1, train_imgs, model)
        model.fit(train_imgs, train_result, batch_size=4, epochs=10, verbose=1, shuffle=True)
        save_intermediate_layer_output(model, train_imgs, ['my_dense_1', 'my_dense_2'], 'lab', 1)

    print('-' * 30)
    print(' Loding Test Image Set')
    print('-' * 30)


def train_and_predict_selected_emotion():
    data_path = 'music_preprocessing'

    # 확인할 이미지 리스트 선정
    emotion_list = ['happy', 'sad']

    save_train_img_file_name = "train_image.npy"
    save_train_result_file_name = "train_result.npy"

    save_test_img_file_name = "test_image.npy"
    save_test_result_file_name = "test_result.npy"

    print('-' * 30)
    print(' Loding Train Image Set')
    print('-' * 30)

    train_imgs_list, train_result_list = load_emotion_select_data(data_path, emotion_list, save_train_img_file_name, save_train_result_file_name)
    # for i in range(len(train_imgs_list)):
    #    train_imgs_list[i] = pre_process(train_imgs_list[i])

    model = small_vgg_19()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-' * 30)
    print(' Fitting Model')
    print('-' * 30)

    for i in range(len(emotion_list) * 10):
        model.fit(train_imgs_list[i % len(emotion_list)], train_result_list[i % len(emotion_list)], batch_size=4, epochs=10, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        # model.fit(train_imgs_list[i], train_result_list[i], batch_size=1, epochs=1000, verbose=1, shuffle=True)

    print('-' * 30)
    print(' Loding Test Image Set')
    print('-' * 30)

    test_imgs_list, test_result_list = load_emotion_select_data(data_path, emotion_list, save_test_img_file_name, save_test_result_file_name)

    print('-' * 30)
    print(' Run Model For Testing')
    print('-' * 30)
    for i in range(len(emotion_list)):
        predict_result = model.predict(test_imgs_list[i])

        print('-' * 30)
        print(' Evaluate Accuracy - ' + emotion_list[i])
        print('-' * 30)
        test_arg_max = np.argmax(test_result_list[i], axis=1)
        predict_arg_max = np.argmax(predict_result, axis=1)

        total_len = test_imgs_list[i].shape[0]

        result = np.equal(test_arg_max, predict_arg_max)

        print(' Test Set Accuracy(' + emotion_list[i] + '): ' + str(np.sum(result) / total_len))


if __name__ == "__main__":
    # train_and_predict()
    train_and_save_result()
