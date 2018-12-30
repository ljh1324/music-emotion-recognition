import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

import array
import math
from pydub.utils import get_array_type


emotion_number = 4
emotion_mapping = {
    'sad': 0,
    'calm': 1,
    'exciting': 2,
    'angry' : 3,
}

data_rows = 512
data_cols = 512


def emotion_to_one_hot_encoding(emotion):
    emotion = emotion_mapping[emotion]
    one_hot = np.zeros(emotion_number)
    one_hot[emotion] = 1
    return one_hot


def extract_librosa_feature_divided_emotion(parent_dir, emotion_dirs, offset, num_of_music, save_info):

    emotion_librosa_feature = []
    emotion_result = []

    f = open(os.path.join(parent_dir, 'librosa_error.txt'), 'w')

    for emotion_dir in emotion_dirs:
        data_path = os.path.join(parent_dir, emotion_dir)
        mp3_list = os.listdir(data_path)

        # 초기화
        idx = 0
        mp3_count = 0
        one_hot_emotion = emotion_to_one_hot_encoding(emotion_dir)

        for mp3 in mp3_list:
            if idx < offset:
                idx += 1
                continue
            items = mp3.split('.')
            if items[len(items) - 1] == 'mp3':

                feature = extract_feature(os.path.join(data_path, mp3))
                if feature.shape != (data_rows, data_cols):
                    print(emotion_dir + ' - ' + mp3)
                    f.write(emotion_dir + ' - ' + mp3 + '\n')
                    continue

                # 저장 형태 지정
                emotion_librosa_feature.append(feature)
                emotion_result.append(one_hot_emotion)

                if mp3_count % 10 == 0:
                    print("Processing " + emotion_dir + " {} / {} done".format(mp3_count, num_of_music))
                mp3_count += 1

                if mp3_count >= num_of_music:
                    emotion_librosa_feature_numpy = np.array(emotion_librosa_feature)
                    np.save(os.path.join(parent_dir, save_info + '_' + emotion_dir + '.npy'), emotion_librosa_feature_numpy)
                    print(emotion_librosa_feature_numpy.shape)
                    emotion_librosa_feature = []

                    emotion_result_numpy = np.array(emotion_result)
                    print(emotion_result_numpy.shape)
                    np.save(os.path.join(parent_dir, save_info + '_' + emotion_dir + '_result.npy'), emotion_result_numpy)
                    emotion_result = []
                    break
        # end for
    # end out for
    f.close()


def extract_librosa_feature(parent_dir, emotion_dirs, offset, num_of_music, save_info):

    emotion_librosa_feature = []
    emotion_result = []

    f = open(os.path.join(parent_dir, 'librosa_error.txt'), 'w')

    for emotion_dir in emotion_dirs:
        data_path = os.path.join(parent_dir, emotion_dir)
        mp3_list = os.listdir(data_path)

        # 초기화
        idx = 0
        mp3_count = 0
        one_hot_emotion = emotion_to_one_hot_encoding(emotion_dir)

        for mp3 in mp3_list:
            if idx < offset:
                idx += 1
                continue
            items = mp3.split('.')
            if items[len(items) - 1] == 'mp3':

                feature = extract_feature(os.path.join(data_path, mp3))
                if feature.shape != (data_rows, data_cols):
                    print(emotion_dir + ' - ' + mp3)
                    f.write(emotion_dir + ' - ' + mp3 + '\n')
                    continue

                # 저장 형태 지정
                emotion_librosa_feature.append(feature)
                emotion_result.append(one_hot_emotion)

                if mp3_count % 10 == 0:
                    print("Processing " + emotion_dir + " {} / {} done".format(mp3_count, num_of_music))
                mp3_count += 1

                if mp3_count >= num_of_music:
                    break
        # end for
    # end out for

    emotion_librosa_feature_numpy = np.array(emotion_librosa_feature)
    np.save(os.path.join(parent_dir, save_info + '.npy'), emotion_librosa_feature_numpy)
    print(emotion_librosa_feature_numpy.shape)

    emotion_result_numpy = np.array(emotion_result)
    print(emotion_result_numpy.shape)
    np.save(os.path.join(parent_dir, save_info + '_result.npy'), emotion_result_numpy)

    f.close()


def extract_librosa_feature_shuffle(parent_dir, emotion_dirs, offset, num_of_music, num_of_test_music,
                                    save_train_file_info, save_test_file_info):

    emotion_librosa_feature = []
    emotion_result = []

    emotion_librosa_feature_for_test = []
    emotion_result_for_test = []

    f = open(os.path.join(parent_dir, 'librosa_error.txt'), 'w')

    for emotion_dir in emotion_dirs:
        print(emotion_dir)
        data_path = os.path.join(parent_dir, emotion_dir)
        mp3_list = os.listdir(data_path)
        np.random.shuffle(mp3_list)

        # 초기화
        idx = 0
        mp3_count = 0
        one_hot_emotion = emotion_to_one_hot_encoding(emotion_dir)

        for mp3 in mp3_list:
            if idx < offset:
                idx += 1
                continue
            items = mp3.split('.')
            if items[len(items) - 1] == 'mp3':
                feature = extract_feature(os.path.join(data_path, mp3))
                if feature.shape != (data_rows, data_cols):
                    print(emotion_dir + ' - ' + mp3)
                    f.write(emotion_dir + ' - ' + mp3 + '\n')
                    continue

                if mp3_count >= num_of_music:
                    emotion_librosa_feature_for_test.append(feature)
                    emotion_result_for_test.append(one_hot_emotion)
                else:
                    emotion_librosa_feature.append(feature)
                    emotion_result.append(one_hot_emotion)

                if mp3_count % 10 == 0:
                    print("Processing " + emotion_dir + " {} / {} done".format(mp3_count, num_of_music))
                mp3_count += 1

                if mp3_count >= num_of_music + num_of_test_music:
                    break
        # end for
    # end out for

    emotion_librosa_feature_numpy = np.array(emotion_librosa_feature)
    np.save(os.path.join(parent_dir, save_train_file_info + '.npy'), emotion_librosa_feature_numpy)
    print(emotion_librosa_feature_numpy.shape)

    emotion_result_numpy = np.array(emotion_result)
    print(emotion_result_numpy.shape)
    np.save(os.path.join(parent_dir, save_train_file_info + '_result.npy'), emotion_result_numpy)

    emotion_librosa_feature_numpy = np.array(emotion_librosa_feature_for_test)
    np.save(os.path.join(parent_dir, save_test_file_info + '.npy'), emotion_librosa_feature_numpy)
    print(emotion_librosa_feature_numpy.shape)

    emotion_result_numpy = np.array(emotion_result_for_test)
    print(emotion_result_numpy.shape)
    np.save(os.path.join(parent_dir, save_test_file_info + '_result.npy'), emotion_result_numpy)

    f.close()


def extract_feature(songname):
    y, sr = librosa.load(songname, duration=60)
    S = np.abs(librosa.stft(y))

    # Extracting Features
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    # chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    # chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    # melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    rmse = librosa.feature.rmse(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    # spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # poly_features = librosa.feature.poly_features(S=S, sr=sr)
    # tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    # zcr = librosa.feature.zero_crossing_rate(y)
    harmonic = librosa.effects.harmonic(y)
    # percussive = librosa.effects.percussive(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)

    # onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    # frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

    flatten_arr = np.array([tempo])
    flatten_arr = np.append(flatten_arr, np.sum(beats.flatten()))
    # flatten_arr = np.append(flatten_arr, chroma_cens.flatten())
    flatten_arr = np.append(flatten_arr, chroma_stft.flatten())
    # print(chroma_stft.shape)
    # flatten_arr = np.append(flatten_arr, chroma_cq.flatten())
    # flatten_arr = np.append(flatten_arr, melspectrogram.flatten())
    # print(melspectrogram.shape)
    flatten_arr = np.append(flatten_arr, rmse.flatten())
    # print(rmse.shape)
    flatten_arr = np.append(flatten_arr, cent.flatten())
    # print(cent.shape)
    # flatten_arr = np.append(flatten_arr, spec_bw.flatten())
    flatten_arr = np.append(flatten_arr, contrast.flatten())
    # print(contrast.shape)
    flatten_arr = np.append(flatten_arr, rolloff.flatten())
    # flatten_arr = np.append(flatten_arr, poly_features.flatten())
    # flatten_arr = np.append(flatten_arr, tonnetz.flatten())
    # flatten_arr = np.append(flatten_arr, zcr.flatten())
    flatten_arr = np.append(flatten_arr, harmonic.flatten())
    print(flatten_arr.shape)
    # flatten_arr = np.append(flatten_arr, percussive.flatten())
    flatten_arr = np.append(flatten_arr, mfcc_delta.flatten())
    # flatten_arr = np.append(flatten_arr, frames_to_time.flatten())

    flatten_arr = flatten_arr[:data_rows * data_cols]
    flatten_arr = np.reshape(flatten_arr, (data_rows, data_cols))

    return flatten_arr


def load_data(parent_dir, save_info, save_result_info, emotion):
    return np.load(os.path.join(parent_dir, save_info + '_' + emotion + '.npy')), np.load(os.path.join(parent_dir, save_result_info + '_' + emotion + '_result.npy'))


def load_total_data(parent_dir, save_file_name, save_result_file_name):
    return np.load(os.path.join(parent_dir, save_file_name)), np.load(os.path.join(parent_dir, save_result_file_name))


def load_test_data(parent_dir, save_info, save_result_info):
    return np.load(os.path.join(parent_dir, save_info + '.npy')), np.load(os.path.join(parent_dir, save_result_info + '.npy'))


if __name__ == '__main__':
    my_parent_dir = 'selected_music'
    my_emotion_dirs = ["sad", "calm", "exciting", "angry"]
    # extract_librosa_feature(parent_dir=my_parent_dir, emotion_dirs=my_emotion_dirs, offset=0, num_of_music=125, save_info='train_selected_librosa_7emotion')
    # extract_librosa_feature(parent_dir=my_parent_dir, emotion_dirs=my_emotion_dirs, offset=125, num_of_music=25, save_info='test_selected_librosa_7emotion')
    extract_librosa_feature_shuffle(parent_dir=my_parent_dir, emotion_dirs=my_emotion_dirs, offset=0, num_of_music=125,
                                    num_of_test_music=25, save_train_file_info='train_selected_librosa_shuffle',
                                    save_test_file_info='test_selected_librosa_shuffle')