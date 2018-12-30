import numpy as np
import os
import array
from scipy.fftpack import fft
from pydub.utils import get_array_type
from pydub import AudioSegment


def mp3_file_name_segment(file_name):
    str_len = len(file_name) - 4    # '.mp3' 부분을 빼기 위해서

    item = ""
    track_name = ""
    flag = False
    for i in range(str_len):
        if file_name[str_len - i - 1] == '-' and not flag:
            track_name = item.strip()
            item = ""
            flag = True
        else:
            item = file_name[str_len - i - 1] + item
    artist_name = item.strip()

    return artist_name, track_name


# 미분
def derivative(arr):
    derivative_arr = []
    temp_arr = np.append(arr, arr[0])
    for i in range(len(temp_arr) - 1):
        derivative_arr.append(temp_arr[i + 1] - temp_arr[i])
    derivative_arr = np.array(derivative_arr)
    return np.array(derivative_arr)


# YouTube에서 추출한 음악의 샘플링 레이트는 48000Hz
# 일반적인 mp3의 샘플링 레이트는 44100Hz
def extract_fft_3d_over_rap(song_name, split_row, split_col, sample_rate):
    start = 30 * sample_rate
    end = 80 * sample_rate
    over_rap_dur = int((end - start) / split_row / 4)

    song = AudioSegment.from_mp3(song_name)
    left = song.split_to_mono()[0]
    bit_depth = left.sample_width * 8
    array_type = get_array_type(bit_depth)
    song = array.array(array_type, left._data)

    song = song[start:end]

    row_item_size = int(len(song) / split_row)

    fft_feature = []
    for i in range(split_row):
        song_fft = np.fft.fft(song[i * row_item_size:(i + 1) * row_item_size + over_rap_dur])
        col_item_size = int(len(song_fft[:int(len(song_fft) / 2)]) / split_col)
        for j in range(split_col):
            item = song_fft[j * col_item_size:(j + 1) * col_item_size]
            item = np.abs(item)
            sum = 0
            for k in range(len(item)):
                sum += item[k].real
            fft_feature.append(sum)

        """
        song_fft = np.fft.fft(song[i * row_item_size:(i + 1) * row_item_size], split_col * 2)
        song_fft = song_fft[:split_col]
        song_fft = np.abs(song_fft)
        fft_feature.append(song_fft[:split_col])
        """

    fft_feature = (fft_feature - np.mean(fft_feature)) / np.std(fft_feature)

    fft_feature = np.array(fft_feature)
    dev_feature = derivative(fft_feature.flatten())
    dev_feature2 = derivative(dev_feature)

    fft_feature = np.reshape(fft_feature, (split_row, split_col))
    dev_feature = np.reshape(dev_feature, (split_row, split_col))
    dev_feature2 = np.reshape(dev_feature2, (split_row, split_col))

    feature = np.append(fft_feature, dev_feature)
    feature = np.append(feature, dev_feature2)
    feature = np.reshape(feature, (split_row, split_col, 3))

    return feature