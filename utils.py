import scipy.io as sio
import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.init as init

from siamese import SiameseNetwork, ContrastiveLoss, Siamese_dropout, Siamese_dropout_hidden, Siamese_st16, Siamese_st8, Siamese_fc, Siamese_Conv, Siamese_Conv_fc
from siamese import Siamese_Conv1_dropout, Siamese_Conv2_dropout

from draw import show_wav

def audio_visual(audio_wav, phoneme_org, SR, vowels):
    phoneme_segs = read_phoneme(phoneme_org)
    audio_segs = audio_seg(audio_wav, phoneme_segs, SR)

    for seg in audio_segs:
        phoneme = seg[1]

        if phoneme in vowels:
            audio = seg[0]
            start = float(seg[2]) / SR
            end = float(seg[3]) / SR
            num_samples = len(audio)

            # print(f"phoneme: {phoneme}")
            # print(f"start: {start}")
            # print(f"end: {end}")
            show_wav(audio, start, end, num_samples)

# For data stats
def stats_single(train_data, test_data):
    phoneme_len = []
    vowel_len = []

    for data in train_data:
        wav_f, phoneme_f, text_f, speaker_dir, rcs = data

        with open(text_f, 'r') as f:
            text = f.read()
            start, end, *_ = text.split()

            len_samples = int(end) - int(start)
            phoneme_len.append(len_samples)

        len_samples_1 = sum(1 for value in rcs if value != 0)
        len_vowels_1 = len_samples_1 // 16
        vowel_len.append(len_vowels_1)
    
    for data in test_data:
        wav_f, phoneme_f, text_f, speaker_dir, rcs = data

        with open(text_f, 'r') as f:
            text = f.read()
            start, end, *_ = text.split()

            len_samples = int(end) - int(start)
            phoneme_len.append(len_samples)

        len_samples_1 = sum(1 for value in rcs if value != 0)
        len_vowels_1 = len_samples_1 // 16
        vowel_len.append(len_vowels_1)

    utterance_len = len(vowel_len)
    phoneme_avg = sum(phoneme_len) / utterance_len
    vowel_avg = sum(vowel_len) / utterance_len

    print(f"For total number of {utterance_len} utterances")
    print(f"Average length of an utterance in samples: {phoneme_avg}")
    print(f"Average length of an utterance in seconds: {phoneme_avg / 16000}")
    print(f"Average number of vowels processed: {vowel_avg}")

# For data stats
def stats_agg(train_data, test_data):
    phoneme_len = []
    vowel_len = []

    for data in train_data:
        speaker1, rcs1, speaker2, rcs2, label = data

        len_samples_1 = sum(1 for value in rcs1 if value != 0)
        len_samples_2 = sum(1 for value in rcs2 if value != 0)
        len_vowels_1 = len_samples_1 // 16
        len_vowels_2 = len_samples_2 // 16
        vowel_len.append(len_vowels_1)
        vowel_len.append(len_vowels_2)
    
    for data in test_data:
        speaker1, rcs1, speaker2, rcs2, label = data

        len_samples_1 = sum(1 for value in rcs1 if value != 0)
        len_samples_2 = sum(1 for value in rcs2 if value != 0)
        len_vowels_1 = len_samples_1 // 16
        len_vowels_2 = len_samples_2 // 16
        vowel_len.append(len_vowels_1)
        vowel_len.append(len_vowels_2)

    utterance_len = len(vowel_len)
    phoneme_avg = sum(phoneme_len) / utterance_len
    vowel_avg = sum(vowel_len) / utterance_len

    print(f"For total number of {utterance_len} utterances")
    print(f"Average number of vowels processed: {vowel_avg}")

# Iterative Data stats for aggregated datasets
def stats_agg_mult(train_tests_paired):
    for train_test_paired in train_tests_paired:
        num_agg, train, test = train_test_paired

        print(f"For agg {num_agg}")
        stats_agg(train, test)

# Load existing datasets
def load_dataset(train_f, test_f):
    # Load datasets
    with open(train_f, 'rb') as f:
        dataset_train_single = pickle.load(f)
        print(f"Loaded {train_f}")
        print(f"Train Dataset length: {len(dataset_train_single)}")
        # for i in range(len(dataset_train_single)):
        #     print(f"phoneme: {dataset_train_single.phonemes[i]}")
        #     print(f"speaker: {dataset_train_single.speakers[i]}")
        #     print(f"text: {dataset_train_single.texts[i]}")
        
    with open(test_f, 'rb') as f:
        dataset_test_single = pickle.load(f)
        print(f"Loaded {test_f}")
        print(f"Test Dataset length: {len(dataset_test_single)}")
    
    return (dataset_train_single, dataset_test_single)

# Load multiple dataset (for agg_sets)
def load_datasets(num_aggs):
    train_tests = []

    for num in num_aggs:
        train_f = f"data/processed/train_agg_{num}.pkl"
        test_f = f"data/processed/test_agg_{num}.pkl"
        # print(train_f)
        # print(test_f)

        with open(train_f, 'rb') as f:
            dataset_train = pickle.load(f)
            # print(f"Loaded {train_f}")
            # print(f"Train Dataset length: {len(dataset_train)}")
        
        with open(test_f, 'rb') as f:
            dataset_test = pickle.load(f)
            # print(f"Loaded {test_f}")
            # print(f"Test Dataset length: {len(dataset_test)}")

        train_tests.append((num, dataset_train, dataset_test))
    
    return train_tests

# Load multiple balanced dataset (for balanced aggs)
def load_datasets_b(num_aggs):
    train_tests = []

    for num in num_aggs:
        train_f = f"data/processed/train_agg_{num}_b.pkl"
        test_f = f"data/processed/test_agg_{num}_b.pkl"

        with open(train_f, 'rb') as f:
            dataset_train = pickle.load(f)
            print(f"Loaded {train_f}")
            print(f"Train Dataset length: {len(dataset_train)}")
        
        with open(test_f, 'rb') as f:
            dataset_test = pickle.load(f)
            print(f"Loaded {test_f}")
            print(f"Test Dataset length: {len(dataset_test)}")

        train_tests.append((num, dataset_train, dataset_test))
    
    return train_tests

# extract train, test pair corresponding to the chosen agg_num from trian_tests
def get_train_test(train_tests, num_agg):
    # num_agg value into index
    if num_agg == 2:
        num_agg = 0
    elif num_agg == 5:
        num_agg = 1
    elif num_agg == 10:
        num_agg = 2

    num, dataset_train, dataset_test = train_tests[num_agg]

    return (num, dataset_train, dataset_test)

# save datasets
def save_dataset(train_src, test_src, train_dst, test_dst):
    with open(train_dst, 'wb') as f:
        pickle.dump(train_src, f)
        print(f"Saved {train_dst}")
    with open(test_dst, 'wb') as f:
        pickle.dump(test_src, f)
        print(f"Saved {test_dst}")

# Extract speaker from text file name
def speaker_from_text(text_f):
    speaker = text_f.split('/')[3]
    return speaker

# Batch-modified
# segment audio file given phoneme labels. Return the segmented audio, corresponding phoneme labels, and the start and end time of each segment
def audio_seg(audio_wav, phoneme_seg):
    audio_seg_list = []

    for audio, phoneme in zip(audio_wav, phoneme_seg):

        rate, y = sio.wavfile.read(audio)
        y = np.array(y, dtype=np.float32)
        # print(f"y.shape: {y.shape}")
        
        audio_seg = []
        for seg in phoneme:
            start = int(float(seg[0]))
            end = int(float(seg[1]))
            phoneme = seg[2]

            audio_seg.append([y[start:end], phoneme, start, end])
        
        audio_seg_list.append(audio_seg)
    
    return audio_seg_list

# Not batch-friendly, used for data preprocessing
def audio_seg_pp(audio_wav, phoneme_seg):
    rate, y = sio.wavfile.read(audio_wav)
    y = np.array(y).astype(float)
    # print(f"y.shape: {y.shape}")

    audio_seg = []
    for seg in phoneme_seg:
        start = int(float(seg[0]))
        end = int(float(seg[1]))
        phoneme = seg[2]

        audio_seg.append([y[start:end], phoneme, start, end])

    return audio_seg

# Batch-modified
# Process the phoneme label file and return the array [# segments, 3], where the column is [start time, end time, phoneme]
def read_phoneme(phoneme_org):
    segs_list = []
    for phoneme in phoneme_org:
        # print(f"phoneme_org: {phoneme_org}")
        # print(f"phoneme: {phoneme}")
        segs = []
        with open(phoneme, 'r') as f:
            for line in f:
                segs.append(line.split())
        segs = np.array(segs)
        segs[:, 0] = segs[:, 0].astype(float)
        segs[:, 1] = segs[:, 1].astype(float)
        # print(f"segs.shape: {segs.shape}")
        # print(f"segs: {segs}")
        segs_list.append(segs)
    return np.array(segs_list)

# Not batch-friendly, used for data preprocessing
def read_phoneme_pp(phoneme_org):
    segs = []
    with open(phoneme_org, 'r') as f:
        for line in f:
            segs.append(line.split())
    segs = np.array(segs)
    segs[:, 0] = segs[:, 0].astype(float)
    segs[:, 1] = segs[:, 1].astype(float)
    # print(f"segs.shape: {segs.shape}")
    # print(f"segs: {segs}")
    return segs

# define siamese networks and return them
def siamese_models():
    siamese_network = SiameseNetwork()
    siamese_dropout = Siamese_dropout()
    siamese_dropout_hiddel = Siamese_dropout_hidden()
    siamese_st16 = Siamese_st16()
    siamese_st8 = Siamese_st8()
    siamese_fc = Siamese_fc()
    siamese_conv = Siamese_Conv()
    siamese_conv_fc = Siamese_Conv_fc()
    
    return [siamese_network, siamese_dropout, siamese_dropout_hiddel, siamese_st16, siamese_st8, siamese_fc, siamese_conv, siamese_conv_fc]

def siamese_models_conv_dropout():
    siamese_conv1_dropout = Siamese_Conv1_dropout()
    siamese_conv2_dropout = Siamese_Conv2_dropout()

    return [siamese_conv1_dropout, siamese_conv2_dropout]

# reset model parameters
def reset_model_params(model):
    dic = model.state_dict()
    for k in dic:
        dic[k] *= 0
    model.load_state_dict(dic)
    del(dic)

def reinitialize_model(model):
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            # Kaiming (He) initialization for Conv1d layers
            init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            # Kaiming (He) initialization for Linear layers
            init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)