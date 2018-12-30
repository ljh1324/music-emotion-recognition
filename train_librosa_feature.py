from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.optimizers import Adam

from music_project.extract_librosa_feauture import load_data, load_test_data, load_total_data

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


def small_vgg_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(512, 512, 1)))
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
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
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

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.000-1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

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
    data_path = 'selected_music'

    save_train_info = "train_selected_librosa"
    save_train_result_info = "train_selected_librosa"

    save_test_info = "test_selected_librosa"
    save_test_result_info = "test_selected_librosa"

    my_emotion_list = ["sad", "happy", "calm", "exciting", "angry"]

    print('-' * 30)
    print(' Loding Train Image Set')
    print('-' * 30)

    train_imgs, train_result = load_data(data_path, save_train_info, save_train_result_info, my_emotion_list[0])
    train_imgs2, train_result2 = load_data(data_path, save_train_info, save_train_result_info, my_emotion_list[1])
    train_imgs3, train_result3 = load_data(data_path, save_train_info, save_train_result_info, my_emotion_list[2])
    train_imgs4, train_result4 = load_data(data_path, save_train_info, save_train_result_info, my_emotion_list[3])
    train_imgs5, train_result5 = load_data(data_path, save_train_info, save_train_result_info, my_emotion_list[4])

    print(train_imgs.shape)
    print(train_result.shape)
    print(train_imgs2.shape)
    print(train_result2.shape)
    print(train_imgs3.shape)
    print(train_result4.shape)
    print(train_imgs4.shape)
    print(train_result4.shape)
    print(train_imgs5.shape)
    print(train_result5.shape)

    train_imgs = train_imgs.astype('float32')
    train_imgs = pre_process(train_imgs)
    train_imgs2 = train_imgs2.astype('float32')
    train_imgs2 = pre_process(train_imgs2)
    train_imgs3 = train_imgs3.astype('float32')
    train_imgs3 = pre_process(train_imgs3)
    train_imgs4 = train_imgs4.astype('float32')
    train_imgs4 = pre_process(train_imgs4)
    train_imgs5 = train_imgs5.astype('float32')
    train_imgs5 = pre_process(train_imgs5)

    """
    mean = np.mean(train_imgs)  # mean for data centering
    std = np.std(train_imgs)  # std for data normalization

    train_imgs -= mean  # nomalization
    train_imgs /= std
    """

    model = small_vgg_19()
    model_checkpoint = ModelCheckpoint('weights_librosa.h5', monitor='val_loss', save_best_only=True)

    print('-' * 30)
    print(' Fitting Model')
    print('-' * 30)

#    model.load_weights('weights.h5')
    for i in range(10000):
        model.fit(train_imgs, train_result, batch_size=4, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        model.fit(train_imgs2, train_result2, batch_size=4, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        model.fit(train_imgs3, train_result3, batch_size=4, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        model.fit(train_imgs4, train_result4, batch_size=4, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        model.fit(train_imgs5, train_result5, batch_size=4, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        model.save_weights('weights_librosa.h5')
    print('-' * 30)
    print(' Loding Test Image Set')
    print('-' * 30)

    test_imgs, test_result = load_test_data(data_path, save_test_info, save_test_result_info)


    test_imgs = test_imgs.astype('float32')
    test_imgs = pre_process(test_imgs)

    """
    mean = np.mean(test_imgs)  # mean for data centering
    std = np.std(test_imgs)  # std for data normalization

    test_imgs -= mean  # nomalization
    test_imgs /= std
    """

    print('-' * 30)
    print(my_emotion_list[0] + ' Run Model')
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

    # print('')
    # print(' Print Loss And Accuracy')

    """
    metrics = model.evaluate(test_imgs, test_result)
    print(my_emotion_list[0] + 'test data results: ')
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
    """


def train_and_predict_for_total():
    data_path = 'selected_music'

    save_train_file_name = "train_selected_librosa_5emotion.npy"
    save_train_result_file_name = "train_selected_librosa_5emotion_result.npy"

    save_test_file_name = "test_selected_librosa_5emotion.npy"
    save_test_result_file_name = "test_selected_librosa_5emotion_result.npy"

    my_emotion_list = ["sad", "happy", "calm", "exciting", "angry"]

    print('-' * 30)
    print(' Loding Train Image Set')
    print('-' * 30)

    train_imgs, train_result = load_total_data(data_path, save_train_file_name, save_train_result_file_name)

    print(train_imgs.shape)
    print(train_result.shape)

    train_imgs = train_imgs.astype('float32')
    train_imgs = pre_process(train_imgs)

    """
    mean = np.mean(train_imgs)  # mean for data centering
    std = np.std(train_imgs)  # std for data normalization

    train_imgs -= mean  # nomalization
    train_imgs /= std
    """

    model = small_vgg_19()
    model_checkpoint = ModelCheckpoint('weights_librosa.h5', monitor='val_loss', save_best_only=True)

    print('-' * 30)
    print(' Fitting Model')
    print('-' * 30)

#    model.load_weights('weights.h5')
    for i in range(10000):
        model.fit(train_imgs, train_result, batch_size=4, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        model.save_weights('weights_librosa.h5')
    print('-' * 30)
    print(' Loding Test Image Set')
    print('-' * 30)

    test_imgs, test_result = load_total_data(data_path, save_test_file_name, save_test_result_file_name)

    test_imgs = test_imgs.astype('float32')
    test_imgs = pre_process(test_imgs)

    """
    mean = np.mean(test_imgs)  # mean for data centering
    std = np.std(test_imgs)  # std for data normalization


    test_imgs -= mean  # nomalization
    test_imgs /= std
    """
    print('-' * 30)
    print(my_emotion_list[0] + ' Run Model')
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

    # print('')
    # print(' Print Loss And Accuracy')

    """
    metrics = model.evaluate(test_imgs, test_result)
    print(my_emotion_list[0] + 'test data results: ')
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
    """


if __name__ == "__main__":
    # train_and_predict()
    train_and_predict_for_total()
