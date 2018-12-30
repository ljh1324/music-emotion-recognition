import music_project.pre_processing_module as ppm
import music_project.load_model_module as lmm


def music_classfication_module(file_name):
    split_row = 224
    split_col = 224
    filter_n = 3

    sampling_rate = 44100
    feature = ppm.extract_fft_3d_over_rap(file_name, split_row, split_col, sampling_rate)

    lmm.load_model("new_jh_net", (split_row, split_col, filter_n))
