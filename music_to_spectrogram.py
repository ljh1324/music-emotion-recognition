"""Generate a Spectrogram image for a given WAV audio sample.

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""


import os
import wave

import pylab
import pydub
from pydub import AudioSegment
from PIL import Image

import numpy
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft


def cut_audio(openfile, save_file):
    seconds = 1000
    pydub.AudioSegment.converter = r"D:\ffmpeg\ffmpeg\bin\ffmpeg.exe"
    sound = AudioSegment.from_mp3(openfile)  # for happy_song.mp3
    sound = sound[: seconds * 80]
    sound.export(save_file, format="wav")


def graph_spectrogram(wav_file, save_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(20, 20))
    pylab.subplot(111)
    """
    subplot(m,n,p) divides the current figure into an m-by-n grid and creates axes in the position specified by p.
    MATLAB® numbers subplot positions by row. The first subplot is the first column of the first row, the second subplot is the second column of the first row, and so on.
    If axes exist in the specified position, then this command makes the axes the current axes.

    """
    pylab.specgram(sound_info, Fs=frame_rate)       # sampling 주차수

    # pylab.show()
    pylab.axis('off')
    pylab.savefig(save_file, bbox_inches='tight', pad_inches=0)
    pylab.close()


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)                     # 전체 프레임 읽음
    """
    An audio frame, or sample, contains amplitude (loudness) information at that particular point in time.
    To produce sound, tens of thousands of frames are played in sequence to produce frequencies.
    """
    sound_info = pylab.fromstring(frames, 'Int16')      #  \x01\x02 => [1, 2]
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


def crop_image(file_name):
    img = Image.open(file_name)

    start_x_point = 54
    start_y_point = 3
    end_x_point = 828   # for genie
    end_y_point = 772
    end_x_point = 1603  # for last fm
    end_y_point = 1542  # for last fm

    area = (start_x_point, start_y_point, end_x_point - start_x_point + 1, end_y_point - start_y_point + 1)
    cropped_img = img.crop(area)
    cropped_img.save(file_name)


def music_segmentation(emotion_dir, emotion_segment_dir, emotion_segment_img_dir):
    emotion_str = emotion_dir[emotion_dir.rfind('\\') + 1:]

    error_file_name = os.path.join(os.path.abspath(os.path.join(emotion_dir, '..')), emotion_str + '-error.txt')
    f_error = open(error_file_name, 'w')

    if not os.path.exists(emotion_segment_dir):
        os.makedirs(emotion_segment_dir)

    if not os.path.exists(emotion_segment_img_dir):
        os.makedirs(emotion_segment_img_dir)

    file_list = os.listdir(emotion_dir)
    total_len = len(file_list)
    idx = 0

    for file in file_list:
        if idx % 10 == 0:
            print('Processing {} / {}'.format(idx, total_len))

        file_name = file[:file.rfind('.')]
        file_wav = file_name + '.wav'
        file_png = file_name + '.png'

        try:
            cut_audio(os.path.join(emotion_dir, file), os.path.join(emotion_segment_dir, file_wav))
        except Exception as e:
            print(str(e))
            f_error.write(file_name + ' - cut_audio_error\n')
            continue

        try:
            graph_spectrogram(os.path.join(emotion_segment_dir, file_wav), os.path.join(emotion_segment_img_dir, file_png))
        except Exception as e:
            print(str(e))
            f_error.write(file_name + ' - graph_spectrogram_error\n')
            continue

        try:
            crop_image(os.path.join(emotion_segment_img_dir, file_png))
        except Exception as e:
            print(str(e))
            f_error.write(file_name + '-crop_image_error\n')
            continue

        idx += 1
        if idx >= 100:
            break

    f_error.close()

if __name__ == '__main__':
    data_path = 'genie_music'
    emotion_dir_list = ['몽환적인', '분노', '슬픔', '신나는', '잔잔한', '행복']
    emotion_segment_dir_list = ['몽환적인-SEG', '분노-SEG', '슬픔-SEG', '신나는-SEG', '잔잔한-SEG', '행복-SEG']
    emotion_segment_img_list = ['몽환적인-SEGIMG', '분노-SEGIMG', '슬픔-SEGIMG', '신나는-SEGIMG', '잔잔한-SEGIMG', '행복-SEGIMG']

    data_path = 'music'
    emotion_dir_list = ['angry_top', 'sad_top', 'exciting_top', 'calm_top', 'happy_top']
    emotion_segment_dir_list = ['angry_seg', 'sad-seg', 'exciting-seg', 'calm-seg', 'happy-seg']
    emotion_segment_img_list = ['angry_segimg', 'sad-segimg', 'exciting-segimg', 'calm-segimg', 'happy-segimg']
    for i in range(len(emotion_dir_list)):
        music_segmentation(emotion_dir=os.path.join(data_path, emotion_dir_list[i]),
                           emotion_segment_dir=os.path.join(data_path, emotion_segment_dir_list[i]),
                           emotion_segment_img_dir=os.path.join(data_path, emotion_segment_img_list[i]))

    #wav_file = 'test2_mood_duration.wav'
    #graph_spectrogram(wav_file)