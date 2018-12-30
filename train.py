from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from music_project.data_input import load_data, load_emotion_select_data
from music_project.data_input import image_rows, image_cols
from keras.callbacks import ModelCheckpoint
# from keras import backend as K

import numpy as np

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
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
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

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    if weights_path:
        model.load_weights(weights_path)

    return model


# 축을 하나 늘린다.
def pre_process(imgs):
    # image 개수, image row, image col 만큼 할당
    imgs_p = np.ndarray((imgs.shape[0], image_rows, image_cols), dtype=np.uint8)
    imgs_p = imgs_p[..., np.newaxis]    # 이미지에 새로운 축 추가(vgg19에 맞도록)

    return imgs_p


def train_and_predict():
    data_path = 'music_preprocessing'

    save_train_img_file_name = "train_image.npy"
    save_train_result_file_name = "train_result.npy"

    save_test_img_file_name = "test_image.npy"
    save_test_result_file_name = "test_result.npy"

    print('-' * 30)
    print(' Loding Train Image Set')
    print('-' * 30)

    train_imgs, train_result = load_data(data_path, save_train_img_file_name, save_train_result_file_name)
    train_imgs = train_imgs.astype('float32')
    print(train_imgs.shape)
    print(train_result.shape)

    model = small_vgg_19()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-' * 30)
    print(' Fitting Model')
    print('-' * 30)

    model.load_weights('weights.h5')
    for i in range(10):
        model.fit(train_imgs, train_result, batch_size=4, epochs=10, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        model.save_weights('weights.h5')

    print('-' * 30)
    print(' Loding Test Image Set')
    print('-' * 30)

    test_imgs, test_result = load_data(data_path, save_test_img_file_name, save_test_result_file_name)
    test_imgs = test_imgs.astype('float32')
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
    train_and_predict()
