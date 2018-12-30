from __future__ import print_function

import os
import numpy as np

from skimage.transform import resize
from skimage.io import imsave, imread

# for unet

image_rows = 224
image_cols = 224

emotion_number = 5

emotion_mapping = {
    'happy': 0,
    'sad': 1,
    'angry': 2,
    'calm': 3,
    'exciting': 4
}


def emotion_to_one_hot_encoding(emotion):
    one_hot = np.zeros(emotion_number)
    one_hot[emotion] = 1
    return one_hot


def create_numpy_data(parent_dir, img_dir_list, input_emotion_list, save_img_file_name, save_img_result_file_name):

    all_imgs = 0
    all_imgs_result = 0

    for i in range(len(img_dir_list)):
        file_data_path = os.path.join(parent_dir, img_dir_list[i])       # data_path와 train데이터를 잇는다.
        one_hot_emotion = emotion_to_one_hot_encoding(emotion_mapping[input_emotion_list[i]])  # 나타나야할 값

        images = os.listdir(file_data_path)                    # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.

        total = len(images)

        imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8) # An array object represents a multidimensional, homogeneous array of fixed-size items.
        imgs_result = np.ndarray((total, emotion_number), dtype=np.uint8)

        idx = 0
        print('-'*30)
        print('Creating training images...')
        print('-'*30)

        for image_name in images:
            img = imread(os.path.join(file_data_path, image_name), as_grey=True)
            print(image_name)
            img_resize = resize(img, (image_rows, image_cols, 3), mode='reflect')
            img_resize = np.array([img_resize])               # np.array 형태로 변형.

            imgs[i] = img_resize
            imgs_result[i] = one_hot_emotion
            if idx % 100 == 0:
                print('Done: {0}/{1} images'.format(idx, total))
            idx += 1

        if i == 0:
            all_imgs = imgs
            all_imgs_result = imgs_result
        else:
            all_imgs = np.concatenate((all_imgs, imgs), axis=0)
            all_imgs_result = np.concatenate((all_imgs_result, imgs_result), axis=0)
    print(all_imgs.shape)

    print('Loading done.')

    np.save(os.path.join(parent_dir, save_img_file_name), all_imgs)
    np.save(os.path.join(parent_dir, save_img_result_file_name), all_imgs_result)

    print('Saving to .npy files done.')


def create_numpy_data_divided_by_emotion(parent_dir, img_dir_list, input_emotion_list, save_img_file_name, save_img_result_file_name):
    for i in range(len(img_dir_list)):
        file_data_path = os.path.join(parent_dir, img_dir_list[i])  # data_path와 train데이터를 잇는다.
        one_hot_emotion = emotion_to_one_hot_encoding(emotion_mapping[input_emotion_list[i]])  # 나타나야할 값

        images = os.listdir(file_data_path)  # os.listdir을 이용하면 해당 디렉터리에 있는 파일들의 리스트를 구할 수 있다.

        total = len(images)

        imgs = np.ndarray((total, image_rows, image_cols, 3),
                          dtype=np.uint8)  # An array object represents a multidimensional, homogeneous array of fixed-size items.
        imgs_result = np.ndarray((total, emotion_number), dtype=np.uint8)

        idx = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        for image_name in images:
            img = imread(os.path.join(file_data_path, image_name), as_grey=False)
            img_resize = resize(img, (image_rows, image_cols, 3), mode='reflect')
            img_resize = np.array([img_resize])  # np.array 형태로 변형.

            imgs[i] = img_resize
            imgs_result[i] = one_hot_emotion
            if idx % 100 == 0:
                print('Done: {0}/{1} images'.format(idx, total))
            idx += 1

        print('Loading done.')

        # parent_dir 아래 "감정이름"_"이미지파일 이름" 형태로 저장
        np.save(os.path.join(parent_dir, input_emotion_list[i] + '_' + save_img_file_name), imgs)
        np.save(os.path.join(parent_dir, input_emotion_list[i] + '_' + save_img_result_file_name), imgs_result)

        print('Saving to .npy files done.')


def load_data(parent_dir, save_img_file_name, save_img_result_file_name):
    imgs_data = np.load(os.path.join(parent_dir, save_img_file_name))
    imgs_result = np.load(os.path.join(parent_dir, save_img_result_file_name))

    return imgs_data, imgs_result


def load_emotion_data(parent_dir, emotion, save_img_file_name, save_img_result_file_name):
    imgs_data = np.load(os.path.join(parent_dir, emotion + '_' + save_img_file_name))
    imgs_result = np.load(os.path.join(parent_dir, emotion + '_' + save_img_result_file_name))

    return imgs_data, imgs_result


def load_emotion_select_data(parent_dir, emotion_list, save_img_file_name, save_img_result_file_name):
    selected_imgs_data = []
    selected_imgs_result_data = []

    for i in range(len(emotion_list)):
        imgs_data, imgs_result = load_emotion_data(parent_dir, emotion_list[i], save_img_file_name, save_img_result_file_name)
        selected_imgs_data.append(imgs_data)
        selected_imgs_result_data.append(imgs_result)

    return selected_imgs_data, selected_imgs_result_data


if __name__ == '__main__':
    # for 4 in 1 image
    data_path = 'music_preprocessing'

    train_img_dir_list = ['top_happy_one_image', 'top_sad_one_image', 'top_angry_one_image', 'top_calm_one_image', 'top_exciting_one_image']
    test_img_dir_list = ['test_happy_one_image', 'test_sad_one_image', 'test_angry_one_image', 'test_calm_one_image', 'test_exciting_one_image']
    emotion_list = ['happy', 'sad', 'angry', 'calm', 'exciting']

    save_train_img_file_name = "train_image.npy"
    save_train_result_file_name = "train_result.npy"

    save_test_img_file_name = "test_image.npy"
    save_test_result_file_name = "test_result.npy"

    # for spectogram image
    data_path = 'genie_music'

    train_img_dir_list = ['행복_SEGIMG', '잔잔한-SEGIMG', '분노-SEGIMG', '슬픔-SEGIMG', '신나는-SEGIMG']
    test_img_dir_list = ['행복_SEGIMG_TEST', '잔잔한-SEGIMG_TEST', '분노-SEGIMG_TEST', '슬픔-SEGIMG_TEST', '신나는-SEGIMG_TEST']
    emotion_list = ['행복', '잔잔함', '분노', '슬픔', '신나는']

    save_train_img_file_name = "train_image.npy"
    save_train_result_file_name = "train_result.npy"

    save_test_img_file_name = "test_image.npy"
    save_test_result_file_name = "test_result.npy"

    create_numpy_data(data_path, train_img_dir_list, emotion_list, save_train_img_file_name, save_train_result_file_name)
    create_numpy_data(data_path, test_img_dir_list, emotion_list, save_test_img_file_name, save_test_result_file_name)

    # create_numpy_data_divided_by_emotion(data_path, train_img_dir_list, emotion_list, save_train_img_file_name, save_train_result_file_name)
    # create_numpy_data_divided_by_emotion(data_path, test_img_dir_list, emotion_list, save_test_img_file_name, save_test_result_file_name)
