# 음악을 주면 3등분하여 chromatogram image로 변환시켜 저장한다.

import librosa.display
import librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt
import music_project.music_to_spectrogram as m2s
import os
import numpy as np
import pylab


# extract chromagram
def extract_chromagram_feature(file_name, save_file_name):
    y, sr = librosa.load(file_name)

    # Use an energy (magnitude) spectrum instead of power spectrogram

    s = np.abs(librosa.stft(y))

    # Compute a chromagram from a waveform or power spectrogram.
    chroma = librosa.feature.chroma_stft(S=s, sr=sr)

    # Use a pre-computed power spectrogram with a larger frame
    # s = np.abs(librosa.stft(y, n_fft=4096))**2
    # chroma = librosa.feature.chroma_stft(S=s, sr=sr)    # S: power spectrogram

    # array([[ 0.685,  0.477, ...,  0.961,  0.986],
    # [ 0.674,  0.452, ...,  0.952,  0.926],
    # ...,
    # [ 0.844,  0.575, ...,  0.934,  0.869],
    # [ 0.793,  0.663, ...,  0.964,  0.972]])

    plt.subplot(1, 1, 1)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma)
    plt.tight_layout()
    plt.savefig(save_file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


# extract db feauture
# https://librosa.github.io/librosa/generated/librosa.core.stft.html?highlight=stft#librosa.core.stft
def extract_db_feature(file_name, save_file_name):
    # From time-series input:

    y, sr = librosa.load(file_name)
    s, phase = librosa.magphase(librosa.stft(y=y))

    plt.subplot(1, 1, 1)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(s, ref=np.max))
    plt.tight_layout()
    plt.savefig(save_file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


# http://keunwoochoi.blogspot.kr/2016/03/2.html
# http://librosa.github.io/librosa/generated/librosa.feature.mfcc.html#librosa.feature.mfcc
def extract_mfcc_feature(file_name, save_file_name):
    y, sr = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs)
    plt.tight_layout()
    plt.savefig(save_file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def extract_feature_in_one_image(file_name, save_file_name):
    y, sr = librosa.load(file_name)

    # chroma
    s = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=s, sr=sr)

    # spectral_contrast
    contrast = librosa.feature.spectral_contrast(S=s, sr=sr)

    # db-scaled spectrogram
    s, phase = librosa.magphase(librosa.stft(y=y))
    db = librosa.amplitude_to_db(s, ref=np.max)

    # mfcc
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    plt.subplot(2, 2, 1)
    librosa.display.specshow(chroma)
    plt.subplot(2, 2, 2)
    librosa.display.specshow(contrast)
    plt.subplot(2, 2, 3)
    librosa.display.specshow(db)
    plt.subplot(2, 2, 4)
    librosa.display.specshow(mfccs)

    plt.tight_layout()
    plt.savefig(save_file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def segmentation_mp3(load_file, save_file):
    sound = AudioSegment.from_mp3(load_file)  # for happy_song.mp3
    duration = 60 * 1000
    sound[:duration].export(save_file + '-begin.mp3', format="mp3")
    sound[duration:duration * 2].export(save_file + '-middle.mp3', format="mp3")
    sound[-duration:].export(save_file + '-end.mp3', format="mp3")


def segmentation_mp3_in_duration(load_file, save_file):
    sound = AudioSegment.from_mp3(load_file)  # for happy_song.mp3
    start = 30 * 1000
    duration = 60 * 1000
    sound[start:start + duration].export(save_file, format="wav")


def split_music_and_create_music_image(mp3_file_dir, save_split_mp3_dir, save_image_dir):
    mp3_list = os.listdir(mp3_file_dir)
    for mp3 in mp3_list:
        file = os.path.join(mp3_file_dir, mp3)

        filename = mp3.split('.')[0]

        segmentation_mp3(file, os.path.join(save_split_mp3_dir, filename))

        extract_chromagram_feature(os.path.join(save_split_mp3_dir, filename + '-begin.mp3'),
                        os.path.join(save_image_dir, filename + '-begin.jpg'))
        extract_chromagram_feature(os.path.join(save_split_mp3_dir, filename + '-middle.mp3'),
                        os.path.join(save_image_dir, filename + '-middle.jpg'))
        extract_chromagram_feature(os.path.join(save_split_mp3_dir, filename + '-end.mp3'),
                        os.path.join(save_image_dir, filename + '-end.jpg'))


def create_music_image(mp3_file_dir, save_image_dir):
    mp3_list = os.listdir(mp3_file_dir)
    total = len(mp3_list)
    idx = 0
    for mp3 in mp3_list:
        file = os.path.join(mp3_file_dir, mp3)

        filename = mp3.split('.')[0]

        extract_chromagram_feature(os.path.join(mp3_file_dir, mp3),
                        os.path.join(save_image_dir, filename + '.jpg'))
        idx += 1
        if idx % 100 == 0:
            print("Extract_feature {} / {} Done", idx, total)

        if idx >= 10: # for experiment
            break


def create_music_feature_image_data(mp3_file_dir, split_file_dir, save_image_dir):
    if not os.path.exists(split_file_dir):
        os.makedirs(split_file_dir)
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)

    mp3_list = os.listdir(mp3_file_dir)
    total = len(mp3_list)
    idx = 0

    f = open(os.path.join(save_image_dir, "segment_error.txt"), 'w')
    for mp3 in mp3_list:
        file = os.path.join(mp3_file_dir, mp3)
        items = mp3.split('.');

        filename = items[len(items) - 1]
        wavename = filename + '.wav'

        # segmentation_mp3_in_duration(file, os.path.join(split_file_dir, mp3))
        try:
            segmentation_mp3_in_duration(file, os.path.join(split_file_dir, wavename))
        except Exception as e:
            print(str(e))
            f.write(mp3 + '\n')
        extract_chromagram_feature(os.path.join(split_file_dir, wavename),
                                   os.path.join(save_image_dir, filename + '_chromatogram' + '.jpg'))
        extract_db_feature(os.path.join(split_file_dir, wavename),
                           os.path.join(save_image_dir, filename + '_db' + '.jpg'))
        m2s.graph_spectrogram(os.path.join(split_file_dir, wavename),
                              os.path.join(save_image_dir, filename + "_spectatogram" + ".jpg"))
        idx += 1
        if idx % 10 == 0:
            print("Extract_feature {} / {} Done", idx, total)
    f.close()


def create_music_feature_one_image_data(mp3_file_dir, split_file_dir, save_image_dir):
    if not os.path.exists(split_file_dir):
        os.makedirs(split_file_dir)
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)

    mp3_list = os.listdir(mp3_file_dir)
    total = len(mp3_list)
    idx = 0

    for mp3 in mp3_list:
        file = os.path.join(mp3_file_dir, mp3)
        filename = mp3.split('.')[0]

        try:
            segmentation_mp3_in_duration(file, os.path.join(split_file_dir, mp3))
        except Exception as e:
            print(str(e))
            f.write(mp3 + '\n')

        extract_feature_in_one_image(os.path.join(split_file_dir, mp3),
                                     os.path.join(save_image_dir, filename + '.jpg'))
        idx += 1
        if idx % 10 == 0:
            print("Extract_feature {} / {} Done".format(idx, total))
    f.close()

#split_music_and_create_music_image('music\\happy_song', 'music_preprocessing\\happy_split_song', 'music_preprocessing\\happy_image')

#create_music_image("music\\happy_song", "music_preprocessing\\happy_image")
#create_music_image("music\\exciting_song", "music_preprocessing\\exciting_image")
#create_music_image("music\\calm_song", "music_preprocessing\\calm_image")
#create_music_image("music\\sad_song", "music_preprocessing\\sad_image")
#create_music_image("music\\angry_song", "music_preprocessing\\angry_image")


if __name__ == '__main__':
    #create_music_feature_one_image_data('music\\happy_top', 'segment_music\\happy_top',
    #                                    'music_preprocessing\\top_happy_one_image')
    #create_music_feature_one_image_data('music\\sad_top', 'segment_music\\sad_top',
    #                                    'music_preprocessing\\top_sad_one_image')
    #create_music_feature_one_image_data('music\\exciting_top', 'segment_music\\exciting_top',
    #                                    'music_preprocessing\\top_exciting_one_image')
    #create_music_feature_one_image_data('music\\calm_top', 'segment_music\\calm_top',
    #                                    'music_preprocessing\\top_calm_one_image')
    #create_music_feature_one_image_data('music\\angry_top', 'segment_music\\angry_top',
    #                                    'music_preprocessing\\top_angry_one_image')

    create_music_feature_one_image_data('music\\happy_test', 'segment_music\\happy_test',
                                       'music_preprocessing\\test_happy_one_image')
    create_music_feature_one_image_data('music\\sad_test', 'segment_music\\sad_test',
                                        'music_preprocessing\\test_sad_one_image')
    create_music_feature_one_image_data('music\\exciting_test', 'segment_music\\exciting_test',
                                        'music_preprocessing\\test_exciting_one_image')
    create_music_feature_one_image_data('music\\calm_test', 'segment_music\\calm_test',
                                        'music_preprocessing\\test_calm_one_image')
    create_music_feature_one_image_data('music\\angry_test', 'segment_music\\angry_test',
                                        'music_preprocessing\\test_angry_one_image')

    # ----------------------------------- 정확도 20% ----------------------------------------------
