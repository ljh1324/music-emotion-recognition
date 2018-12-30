import numpy as np
import os
import array
from scipy.fftpack import fft
from pydub.utils import get_array_type
from pydub import AudioSegment


emotion_number = 6
emotion_mapping = {
    'sad': 0,
    'exciting': 1,
    'calm': 2,
    'angry': 3,
    'dreamlike': 4,
    'happy': 5,
}

data_rows = 180
data_cols = 180


def emotion_to_one_hot_encoding(emotion):
    emotion = emotion_mapping[emotion]
    one_hot = np.zeros(emotion_number)
    one_hot[emotion] = 1
    return one_hot


def extract_fft_value(parent_dir, emotion_dirs, offset, num_of_music, save_info):
    start = 50 * 1000     # start 50 second
    end = start + 80 * 1000  # end 82.4 second, 32400 : 180 * 180 for CNN, root(30000) = 173.xx

    emotion_fft = []
    emotion_result = []

    f = open(os.path.join(parent_dir, 'fft_error.txt'), 'w')

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
                song = AudioSegment.from_mp3(os.path.join(data_path, mp3))
                left = song.split_to_mono()[0]
                bit_depth = left.sample_width * 8
                array_type = get_array_type(bit_depth)
                song = array.array(array_type, left._data)

                # start ~ end 까지 추출하여 fft 적용후 비율값 저장
                print(len(song))
                song_fft = np.fft.fft(song)
                print(len(song_fft))
                song_fft= fft(song, 48000)
                print('song_fft:', len(song_fft))
                song = song[start:end]
                print('song_len:', len(song))
                song_fft = np.fft.fft(song)
                print(len(song_fft))
                if True:
                    continue
                song_fft = np.abs(song_fft)
                total = np.sum(song_fft)
                print(total)
                if total < 10000:
                    f.write(emotion_dir + ' - ' + mp3 + '\n')

                song_fft /= total

                # 저장 형태 지정
                song_fft = song_fft.reshape((180, 180))
                emotion_fft.append(song_fft)
                emotion_result.append(one_hot_emotion)

                if mp3_count % 10 == 0:
                    print("Processing " + emotion_dir + " {} / {} done".format(mp3_count, num_of_music))
                mp3_count += 1

                if mp3_count >= num_of_music:
                    break
        idx += 1

    emotion_fft_numpy = np.array(emotion_fft)
    print(emotion_fft_numpy.shape)
    emotion_result_numpy = np.array(emotion_result)
    print(emotion_result_numpy.shape)
    # 부모 디렉토리\\파일명.npy 로 저장
    np.save(os.path.join(parent_dir, save_info + '.npy'), emotion_fft_numpy)
    np.save(os.path.join(parent_dir, save_info + '_result.npy'), emotion_result_numpy)
    f.close()


def derivative(arr):
    derivative_arr = []
    temp_arr = np.append(arr, arr[0])
    for i in range(len(temp_arr) - 1):
        derivative_arr.append(temp_arr[i + 1] - temp_arr[i])
    derivative_arr = np.array(derivative_arr)
    return np.array(derivative_arr)


def extract_fft_1d(song_name, split_row, split_col):
    sample_rate = 48000
    start = 30 * sample_rate
    end = 80 * sample_rate

    song = AudioSegment.from_mp3(song_name)
    left = song.split_to_mono()[0]
    bit_depth = left.sample_width * 8
    array_type = get_array_type(bit_depth)
    song = array.array(array_type, left._data)

    song = song[start:end]

    row_item_size = int(len(song) / split_row)

    fft_feature = []
    for i in range(split_row):
        song_fft = np.fft.fft(song[i * row_item_size:(i + 1) * row_item_size])
        col_item_size = int(len(song_fft[:int(len(song_fft) / 2)]) / split_col)
        for j in range(split_col):
            item = song_fft[j * col_item_size:(j+1) * col_item_size]
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

    feature = np.reshape(fft_feature, (split_row, split_col, 1))

    return feature


def extract_fft_1d_over_rap(song_name, split_row, split_col):
    sample_rate = 48000
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
            item = song_fft[j * col_item_size:(j+1) * col_item_size]
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

    feature = np.reshape(fft_feature, (split_row, split_col, 1))

    return feature


def extract_fft_2d(song_name, split_row, split_col):
    sample_rate = 48000
    start = 30 * sample_rate
    end = 80 * sample_rate

    song = AudioSegment.from_mp3(song_name)
    left = song.split_to_mono()[0]
    bit_depth = left.sample_width * 8
    array_type = get_array_type(bit_depth)
    song = array.array(array_type, left._data)

    song = song[start:end]

    row_item_size = int(len(song) / split_row)

    fft_feature = []
    for i in range(split_row):
        song_fft = np.fft.fft(song[i * row_item_size:(i + 1) * row_item_size])
        col_item_size = int(len(song_fft[:int(len(song_fft) / 2)]) / split_col)
        for j in range(split_col):
            item = song_fft[j * col_item_size:(j+1) * col_item_size]
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

    fft_feature = np.reshape(fft_feature, (split_row, split_col))
    dev_feature = np.reshape(dev_feature, (split_row, split_col))

    feature = np.append(fft_feature, dev_feature)
    feature = np.reshape(feature, (split_row, split_col, 2))

    return feature


def extract_fft_2d_over_rap(song_name, split_row, split_col):
    sample_rate = 48000
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
            item = song_fft[j * col_item_size:(j+1) * col_item_size]
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

    fft_feature = np.reshape(fft_feature, (split_row, split_col))
    dev_feature = np.reshape(dev_feature, (split_row, split_col))

    feature = np.append(fft_feature, dev_feature)
    feature = np.reshape(feature, (split_row, split_col, 2))

    return feature


def extract_fft_3d(song_name, split_row, split_col):
    sample_rate = 48000
    start = 30 * sample_rate
    end = 80 * sample_rate

    song = AudioSegment.from_mp3(song_name)
    left = song.split_to_mono()[0]
    bit_depth = left.sample_width * 8
    array_type = get_array_type(bit_depth)
    song = array.array(array_type, left._data)

    song = song[start:end]

    row_item_size = int(len(song) / split_row)

    fft_feature = []
    for i in range(split_row):
        song_fft = np.fft.fft(song[i * row_item_size:(i + 1) * row_item_size])
        col_item_size = int(len(song_fft[:int(len(song_fft) / 2)]) / split_col)
        for j in range(split_col):
            item = song_fft[j * col_item_size:(j+1) * col_item_size]
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


def extract_fft_3d_over_rap(song_name, split_row, split_col):
    sample_rate = 48000
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


def extract_fft_fix_size(parent_dir, emotion_dirs, offset, num_of_music, num_of_test_music, save_train_file_info, save_test_file_info):
    emotion_feature = []
    emotion_result = []

    emotion_test_feature = []
    emotion_test_result = []

    split_row = 224
    split_col = 224

    f = open(os.path.join(parent_dir, 'fft_error.txt'), 'w')

    for emotion_dir in emotion_dirs:
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
                # feature = extract_fft_3d(os.path.join(data_path, mp3), split_row=split_row, split_col=split_col)
                feature = extract_fft_1d_over_rap(os.path.join(data_path, mp3), split_row=split_row, split_col=split_col)
                if mp3_count >= num_of_music:
                    emotion_test_feature.append(feature)
                    emotion_test_result.append(one_hot_emotion)
                else:
                    emotion_feature.append(feature)
                    emotion_result.append(one_hot_emotion)

                if mp3_count % 10 == 0:
                    print("Processing " + emotion_dir + " {} / {} done".format(mp3_count, num_of_music + num_of_test_music))
                mp3_count += 1

                if mp3_count >= num_of_music + num_of_test_music:
                    break
        idx += 1

    emotion_feature_numpy = np.array(emotion_feature)
    print(emotion_feature_numpy.shape)
    emotion_result_numpy = np.array(emotion_result)
    print(emotion_result_numpy.shape)
    # 부모 디렉토리\\파일명.npy 로 저장
    np.save(os.path.join(parent_dir, save_train_file_info + '.npy'), emotion_feature_numpy)
    np.save(os.path.join(parent_dir, save_train_file_info + '_result.npy'), emotion_result_numpy)

    emotion_feature_numpy = np.array(emotion_test_feature)
    print(emotion_feature_numpy.shape)
    emotion_result_numpy = np.array(emotion_test_result)
    print(emotion_result_numpy.shape)
    # 부모 디렉토리\\파일명.npy 로 저장
    np.save(os.path.join(parent_dir, save_test_file_info + '.npy'), emotion_feature_numpy)
    np.save(os.path.join(parent_dir, save_test_file_info + '_result.npy'), emotion_result_numpy)

    f.close()


def extract_fft_feature(file_dir, offset, save_file_info):
    emotion_feature = []
    emotion_result = []

    emotion_test_feature = []
    emotion_test_result = []

    split_row = 512
    split_col = 512

    f = open(os.path.join(file_dir, 'fft_error.txt'), 'w')

    for emotion_dir in emotion_dirs:
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
                # feature = extract_fft_3d(os.path.join(data_path, mp3), split_row=split_row, split_col=split_col)
                feature = extract_fft_1d_over_rap(os.path.join(data_path, mp3), split_row=split_row, split_col=split_col)

                emotion_test_feature.append(feature)
                emotion_test_result.append(one_hot_emotion)

                if mp3_count % 10 == 0:
                    print("Processing " + emotion_dir + " {} / {} done".format(mp3_count, num_of_music + num_of_test_music))
                mp3_count += 1
        idx += 1

    emotion_feature_numpy = np.array(emotion_feature)
    print(emotion_feature_numpy.shape)
    emotion_result_numpy = np.array(emotion_result)
    print(emotion_result_numpy.shape)
    # 부모 디렉토리\\파일명.npy 로 저장
    np.save(os.path.join(parent_dir, save_file_info + '.npy'), emotion_feature_numpy)
    np.save(os.path.join(parent_dir, save_file_info + '_result.npy'), emotion_result_numpy)

    f.close()


def load_data(parent_dir, save_fft_file, save_result_file):
    return np.load(os.path.join(parent_dir, save_fft_file)), np.load(os.path.join(parent_dir, save_result_file))


if __name__ == '__main__':
    #song = 'test_for_fft.mp3'
    # extract_fft(song, 512, 512)

    """
    my_parent_dir = 'total_genie_music'
    my_emotion_dirs = ["슬픔", "신나는", "잔잔한", "행복", "분노"]

    extract_fft_value(parent_dir=my_parent_dir, emotion_dirs=my_emotion_dirs, offset=0, num_of_music=100, save_info='train')
    extract_fft_value(parent_dir=my_parent_dir, emotion_dirs=my_emotion_dirs, offset=100, num_of_music=20, save_info='test')
    """

    my_parent_dir = 'selected_music'
    my_emotion_dirs = ["sad", "exciting", "calm", "angry", "dreamlike", "happy"]

    extract_fft_fix_size(parent_dir=my_parent_dir, emotion_dirs=my_emotion_dirs, offset=0,
                         num_of_music=125, num_of_test_music=25,
                         save_train_file_info='train_fft_1d_over_rap_6emotion', save_test_file_info='test_fft_1d_over_rap_6emotion')
