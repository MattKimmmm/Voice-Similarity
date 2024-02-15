import numpy as np
from transfer import tf
import os
from itertools import combinations
import time
from numba import njit
import scipy.io as sio
import random

import torch
import pickle

from draw import show_wav, plot_signal
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
from utils import speaker_from_text, read_phoneme, audio_seg, read_phoneme_pp, audio_seg_pp, save_dataset

# Custom Audio Dataset for TIMIT data
class AudioDataset(Dataset):
    def __init__(self, root_dir, rcs_init, epochs, sr, threshold_vc, num_tubes, vowels, offset):
        self.audios = []    # Audio file
        self.speakers = []  # Speaker ID
        self.phonemes = []  # Phoneme file
        self.texts = []     # Text file
        self.rcs = []       # Reflection Coefficients
        self.root_dir = root_dir
        self.rcs_init = rcs_init
        self.epochs = epochs
        self.sr = sr
        self.threshold_vc = threshold_vc
        self.num_tubes = num_tubes
        self.vowels = vowels
        self.offset = offset

        self._load_data()

    def _load_data(self):
        # Create pair of audio files
        print("Loading data...")
        # For multiprocessing
        args_list = []
        count = 0

        # Currently just DR1

        for dialect in os.listdir(self.root_dir):
        #     dialect_dir = os.path.join(self.root_dir, dialect)

            dialect_dir = os.path.join(self.root_dir, dialect)

            # Check if its a directory (not .DS_Store)
            if not os.path.isdir(dialect_dir):
                continue
 
            print(f"Loading data from {dialect} dialect")

            for speaker in os.listdir(dialect_dir):
                # Iterate over files in speaker directory and extract relevant files
                speaker_dir = os.path.join(dialect_dir, speaker)
                # print(f"For speaker {speaker}")

                if not os.path.isdir(speaker_dir):
                    continue

                # For multiprocessing
                args = (speaker_dir, self.rcs_init, self.epochs, self.sr, self.threshold_vc, self.num_tubes, self.vowels, self.offset)
                args_list.append(args)

                # from the directory name extract gender, initials, and index
                # print(f"Loading data from {speaker} speaker")
                gender = speaker[0]
                initials = speaker[1:3]
                index = speaker[4]

        
        with Pool(processes=8) as pool:
            results = pool.map(process_speaker, args_list)

        for results_set in results:
            since = time.time()

            for result in results_set:
                wav_f, phoneme_f, text_f, speaker_dir, rcs = result
                self.phonemes.append(phoneme_f)
                self.texts.append(text_f)
                self.audios.append(wav_f)
                self.speakers.append(speaker)
                self.rcs.append(rcs)
        
            print(f"Single Pool with 8 processes took {time.time() - since}")

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio = self.audios[idx]
        phoneme = self.phonemes[idx]
        text = self.texts[idx]
        speaker = self.speakers[idx]
        rcs = self.rcs[idx]

        return audio, phoneme, text, speaker, rcs
    

# Given a dataset, create pairs of them
class AudioPair(Dataset):
    def __init__(self, audio_dataset):
        self.audio_dataset = audio_dataset
        self.pairs = []
        self.labels = []

        self._create_pairs()

    # Return index combination 
    def _create_pairs(self):
        # Unique paring
        unique_pairs = list(combinations(range(len(self.audio_dataset)), 2))
        # Self paring
        self_pairs = [(i, i) for i in range(len(self.audio_dataset))]

        self.pairs = unique_pairs + self_pairs
        return self.pairs    
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]

        wav_f_1, phoneme_f_1, text_f_1, speaker_dir_1, rcs_1 = self.audio_dataset[idx1]
        wav_f_2, phoneme_f_2, text_f_2, speaker_dir_2, rcs_2 = self.audio_dataset[idx2]
        
        speaker_dir_1 = speaker_from_text(text_f_1)
        speaker_dir_2 = speaker_from_text(text_f_2)

        gender1 = speaker_dir_1[0]
        initials1 = speaker_dir_1[1:3]
        index1 = speaker_dir_1[4]
        gender2 = speaker_dir_2[0]
        initials2 = speaker_dir_2[1:3]
        index2 = speaker_dir_2[4]

        label = 1
        if (initials1 == initials2) and (index1 == index2):
            label = 0
        
        return wav_f_1, phoneme_f_1, text_f_1, speaker_dir_1, rcs_1, wav_f_2, phoneme_f_2, text_f_2, speaker_dir_2, rcs_2, label
    
# Create paired datasets from single datasets
def create_pairs(train_test):
    train_single, test_single = train_test
    train_paired = AudioPair(train_single)
    test_paired = AudioPair(test_single)
    print(f"Paired train dataset length: {len(train_paired)}")
    print(f"Paired train dataset length: {len(test_paired)}")
    return train_paired, test_paired

# For aggregated Audio
class RCSDataset(Dataset):
    def __init__(self, rcs_speaker):
        self.rcs = []
        self.speaker = []
        self.rcs_speaker = rcs_speaker
        self._load_data()

    def __len__(self):
        return len(self.rcs)

    def __getitem__(self, idx):
        speaker = self.speaker[idx]
        rcs = self.rcs[idx]
        return speaker, rcs
    
    def _load_data(self):

        for item in self.rcs_speaker:
            speaker, rcs = item

            self.rcs.append(rcs)
            self.speaker.append(speaker)

class RCSPair(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.pairs = []
        self.labels = []
        self._create_pairs()

    def _create_pairs(self):
        # Unique paring
        unique_pairs = list(combinations(range(len(self.dataset)), 2))
        # Self paring
        self_pairs = [(i, i) for i in range(len(self.dataset))]

        self.pairs = unique_pairs + self_pairs
        return self.pairs    

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]

        speaker1, rcs1 = self.dataset[idx1]
        speaker2, rcs2 = self.dataset[idx2]

        gender1 = speaker1[0]
        initials1 = speaker1[1:3]
        index1 = speaker1[4]
        gender2 = speaker2[0]
        initials2 = speaker2[1:3]
        index2 = speaker2[4]

        label = 1
        if (speaker1 == speaker2):
            label = 0
        
        return speaker1, rcs1, speaker2, rcs2, label
    
# Create RCSpairs for different aggregations
def make_RCSPair(train_tests):
    train_tests_pair = []

    for train_test in train_tests:
        num_agg, train, test = train_test
        train_pair = RCSPair(train)
        test_pair = RCSPair(test)
        print(f"For aggregation of {num_agg}")
        print(f"Paired train dataset length: {len(train_pair)}")
        print(f"Paired test dataset length: {len(test_pair)}")

        train_tests_pair.append((num_agg, train_pair, test_pair))
    
    return train_tests_pair
    
# For multiprocessing
def process_speaker(args):
    since_f = time.time()
    speaker_dir, rcs_init, epochs, sr, threshold_vc, num_tubes, vowels, offset = args

    results = []
    files = sorted(os.listdir(speaker_dir))
    phoneme_f = text_f = wav_f = None

    for file in files:
        if file.endswith(".PHN"):
            phoneme_f = os.path.join(speaker_dir, file)
        elif file.endswith(".TXT"):
            text_f = os.path.join(speaker_dir, file)
        elif file.endswith(".wav"):
            wav_f = os.path.join(speaker_dir, file)

        if phoneme_f and text_f and wav_f:
            since = time.time()
            rcs = audio_single_pp(rcs_init, epochs, sr, threshold_vc, num_tubes, wav_f, phoneme_f, vowels, offset)
            print(f"Single audio took {time.time() - since}s")
            results.append((wav_f, phoneme_f, text_f, speaker_dir, rcs))
            phoneme_f = text_f = wav_f = None

    print(f"A single speaker process took {time.time() - since_f} s")

    return results
    
# Batch-modified
# Takes in a single audio file and phoneme segmentation file and returns the input layer for the network
# 16 * 20 (# vowels) = 320
def audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio_wav, phoneme_seg, vowels, offset):
    # print(phoneme_seg)
    audio_batch = audio_seg(audio_wav, read_phoneme(phoneme_seg))
    rcs_layers = []

    for audios in audio_batch:
        results = []

        for seg in audios:
            audio = seg[0]
            phoneme = seg[1]
            start = seg[2]
            end = seg[3]
            # print(f"phoneme: {phoneme}")
            # print(f"start: {start}")
            # print(f"end: {end}")

            if phoneme in vowels:
                # print(f"For Phoneme: {phoneme}")
                rcs, error = rcs_single(offset, audio, phoneme, rcs, epochs, sr, threshold_vc, num_tubes)
                results.append((phoneme, rcs, error))

        for result in results:
            phoneme, rcs, error = result
            # print(f"phoneme: {phoneme}")
            # print(f"rcs: {rcs}")
            # print(f"error: {error}")
    
        rcs_layer = make_input(results, vowels)
        rcs_layers.append(rcs_layer)

    return np.array(rcs_layers)

# Not Batch-friendly, used for data preprocessing
def audio_single_pp(rcs, epochs, sr, threshold_vc, num_tubes, audio_wav, phoneme_seg, vowels, offset):
    audio_segs = audio_seg_pp(audio_wav, read_phoneme_pp(phoneme_seg))
    results = []
    for seg in audio_segs:
        audio = seg[0]
        phoneme = seg[1]
        start = seg[2]
        end = seg[3]
        # print(f"phoneme: {phoneme}")
        # print(f"start: {start}")
        # print(f"end: {end}")
        if phoneme in vowels:
            # print(f"For Phoneme: {phoneme}")
            rcs, error = rcs_single(offset, audio, phoneme, rcs, epochs, sr, threshold_vc, num_tubes)
            results.append((phoneme, rcs, error))
    for result in results:
        phoneme, rcs, error = result
        # print(f"phoneme: {phoneme}")
        # print(f"rcs: {rcs}")
        # print(f"error: {error}")
    
    rcs_layer = make_input(results, vowels)
    return rcs_layer

# Calculate the frequency response of the input audio
@njit
def calc_fft(audio):
    num_samples = len(audio)
    f_res = np.fft.fft(audio)
    # f_res = np.array(f_res, dtype=np.complex64)
    f_res = f_res[:num_samples // 2]
    f_res = 20 * np.log10(np.abs(f_res))
    # f_res = 20 * np.log10(f_res)
    f_res = f_res - np.mean(f_res)
    # f_res = np.abs(f_res)
    return f_res

# calculate the frequency response of the transfer function
@njit
def f_res_transfer(audio, phoneme, rcs, freqs, num_tubes, sr):
    f_ress = np.empty(len(freqs), dtype=np.float32)

    for i, f in enumerate(freqs):
        f_res = tf(rcs, f, sr, num_tubes)
        f_ress[i] = f_res
    # print(f_ress)

    # float32
    return 20 * np.log10(np.abs(f_ress))

# rcs training loop for a given phoneme
@njit
def rcs_single(offset, audio, phoneme, rcs, epochs, sr, threshold, num_tubes):
    # Variables
    loss_curr = np.float32(0)
    loss_prev = np.inf
    count = np.int32(0)

    # frequency response from original audio = target
    # truncate it to the SR / 2
    num_samples = np.int32(len(audio))
    f_res_org = calc_fft(audio)
    # print(f"len(f_res_org): {len(f_res_org)}")
    # print(f"f_res_org: {f_res_org}")

    # Calculate the corresponding frequency values for original audio
    freq_bin = np.arange(0, num_samples) * sr / num_samples
    freq_bin_pos = freq_bin[:num_samples // 2].astype(np.float32)

    # print("freq_bin len: ", len(freq_bin_pos))
    # print("freq_bin: ", freq_bin_pos)

    for i in range(epochs):
        # Iterate over rcs
        loss_tube = np.empty(len(rcs), dtype=np.float32)
        for j in range(len(rcs)):
            # Add offset to rc
            rcs_up = rcs.copy()
            rcs_down = rcs.copy()
            rcs_up[j] += offset
            rcs_down[j] -= offset
            # print(rcs)

            # frequency response from transfer function for rc_up and rc_down
            # since = time.time()
            f_res_tf_up = f_res_transfer(audio, phoneme, rcs_up, freq_bin_pos, num_tubes, sr)
            # print(f"Time taken for f_res_tf_up: {time.time() - since}s")
            f_res_tf_down = f_res_transfer(audio, phoneme, rcs_down, freq_bin_pos, num_tubes, sr)
            # print(f"len(f_res_tf_up): {len(f_res_tf_up)}")

            # calculate loss
            loss_up = np.sum(np.abs(f_res_org - f_res_tf_up), dtype=np.float32)
            loss_down = np.sum(np.abs(f_res_org - f_res_tf_down), dtype=np.float32)
            # print(f"loss_up: {loss_up}")
            # print(f"loss_down: {loss_down}")

            # update rcs
            if loss_up < loss_down:
                rcs = rcs_up
                loss_curr = loss_up
            else:
                rcs = rcs_down
                loss_curr = loss_down
            
            loss_tube[j] = loss_curr

            # print("Current Offset at: ", offset)
        
        # loss avg over one whole tube
        loss_avg_tube = np.mean(loss_tube)

        if np.abs(loss_avg_tube - loss_prev) < threshold:
            # print("Error improvement less than threshold. Terminating")
            # print(f"Train done at Epoch {i}")
            break

        if loss_avg_tube > loss_prev:
            offset *= 0.5
            loss_prev = loss_curr
            count += 1

            if count > 4:
                # print("Error not improving in 5 consecutive steps. Terminating")
                # print(f"Train done at Epoch {i}")
                break
            
        # when loss < loss_prev
        if loss_avg_tube < loss_prev:
            offset *= 1.01
            loss_prev = loss_curr

            count = 0
        
        loss_prev = loss_avg_tube
        
        # if i % 20 == 0:
        #     print(f"Loss at Epoch {i}: {loss_avg_tube}")
        #     print("Current rcs: ", rcs)

    # print(f"Outputs for {phoneme}:")
    # print(f"Final rcs: {rcs}")
    # print(f"Final loss: {loss_avg_tube}")

    # Plot
    # path = "./figures/paper"
    # title = f"Frequency Response of {phoneme}"
    # plot_signal(freq_bin_pos, f_res_org, path, title, phoneme, True)
    # title = f"V(z) of {phoneme}"
    # plot_signal(freq_bin_pos, f_res_tf_up, path, title, phoneme, False)

    return rcs, loss_avg_tube

# switch for vowels
# Vowel order:
# "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"
def make_input(results, vowels):
    rcs_layer = np.array([])
    vowel_dict = {
        "iy": [], "ih": [], "eh": [], "ey": [], "ae": [], "aa": [], "aw": [], "ay": [], "ah": [], "ao": [], "oy": [], "ow": [], "uh": [], "uw": [], "ux": [], "er": [], "ax": [], "ix": [], "axr": [], "ax-h": []
    }

    # Aggregate rcs for each vowel
    for result in results:
        phoneme, rcs, error = result
        vowel_dict[phoneme].append(rcs)

    # Go over the vowel dict, adds zeros[16] if the list is empty, average rcs if not
    for vowel_item in vowel_dict.items():
        vowel, rcs_list = vowel_item
        num_captured = len(rcs_list)

        if num_captured == 0:
            rcs_list.append(np.zeros(16))
        elif num_captured > 1:
            rcs_list = np.array(rcs_list)
            rcs_list[0] = np.mean(rcs_list)

    # Go over the vowel dict again, and add the rcs to the rcs_layer
    for rcs in vowel_dict.values():
        rcs_layer = np.concatenate((rcs_layer, rcs[0]), axis=None)
    
    return rcs_layer

# Given Audio, print out everything
def audio_single_paper(rcs, epochs, sr, threshold_vc, num_tubes, audio_wav, phoneme_seg, text, vowels, offset):
    audio_segs = audio_seg_pp(audio_wav, read_phoneme_pp(phoneme_seg))

    t = ""
    # get text
    with open(text, 'r') as f:
        for line in f:
            t = line
    print(f"Text: {t}")

    # original audio plot
    start_audio = 0
    end_audio = 50074
    rate, audio_wav = sio.wavfile.read(audio_wav)
    show_wav(audio_wav, start_audio, end_audio, len(audio_wav), "./figures/paper", sr)

    results = []
    for seg in audio_segs:
        audio = seg[0]
        phoneme = seg[1]
        start = seg[2]
        end = seg[3]
        # print(f"phoneme: {phoneme}")
        # print(f"start: {start}")
        # print(f"end: {end}")
        if phoneme in vowels:
            # print(f"For Phoneme: {phoneme}")
            rcs, error = rcs_single(offset, audio, phoneme, rcs, epochs, sr, threshold_vc, num_tubes)
            results.append((phoneme, rcs, error))

    for result in results:
        phoneme, rcs, error = result
        # print(f"phoneme: {phoneme}")
        # print(f"rcs: {rcs}")
        # print(f"error: {error}")
    
    rcs_layer = make_input(results, vowels)
    return rcs_layer

# Agregate audios for a single speaker
def into_full_phoneme(dataset, num_samples):

    rcs_dict = {}
    phoneme_dict = {}
    text_dict = {}

    for data in dataset:
        wav_f, phoneme_f, text_f, speaker_dir, rcs = data
        speaker = speaker_from_text(text_f)

        if speaker in rcs_dict.keys():
            rcs_dict[speaker].append(rcs)
        else:
            rcs_dict[speaker] = []
            rcs_dict[speaker].append(rcs)
        
        if speaker in phoneme_dict.keys():
            phoneme_dict[speaker].append(phoneme_f)
        else:
            phoneme_dict[speaker] = []
            phoneme_dict[speaker].append(phoneme_f)
        
        if speaker in text_dict.keys():
            text_dict[speaker].append(text_f)
        else:
            text_dict[speaker] = []
            text_dict[speaker].append(text_f)

    # iterate over each item and parse the lists (as keys) accordingly with the given number of aggregation
    if num_samples != 1:
        agg_tuple = []
        # Segment each item (of length 10) into aggregated splits
        for rcs_item, phoneme_item, text_item in zip(rcs_dict.items(), phoneme_dict.items(), text_dict.items()):

            speaker, rcs_list = rcs_item
            _, phoneme_list = phoneme_item
            _, text_list = text_item

            arr_len = len(rcs_list)

            rcs_segments = [rcs_list[i:i+num_samples] for i in range(0, arr_len, num_samples)]
            phoneme_segments = [phoneme_list[i:i+num_samples] for i in range(0, arr_len, num_samples)]
            text_segments = [phoneme_list[i:i+num_samples] for i in range(0, arr_len, num_samples)]

            # Add each segment to the existing items, deleting the original items that were used for segmentation
            for rcs_seg, phoneme_seg, text_seg in zip(rcs_segments, phoneme_segments, text_segments):
                agg_tuple.append((speaker, rcs_seg, phoneme_seg, text_seg))
                # print(f"Segment length: {len(rcs_seg)}")
                # print(f"Speaker: {speaker}")

    # print(f"rcs_dict shape: {len(rcs_dict)}")
    # print(f"rcs_dict: {rcs_dict}")
    # print(f"phoneme_dict shape: {len(phoneme_dict)}")
    # print(f"phoneme_dict: {phoneme_dict}")
    # print(f"text_dict shape: {len(text_dict)}")
    # print(f"text_dict: {text_dict}")

    speaker_rcs = []

    for item in agg_tuple:
        speaker, rcs_seg, phoneme_seg, text_seg = item

        rc_list = [[0, 0] for _ in range(320)]
        rc_final = [0] * 320

        # over 320-len rcs from the rcs_seg(size=num_samples)
        for rc_single in rcs_seg:
            # over each rc (single value)
            for i, rc in enumerate(rc_single):
                # Only when rc != 0 adds the rc and increment the count
                if rc != 0:
                    rc_list[i][0] += rc
                    rc_list[i][1] += 1
            
            # Sum over segments and average them. Empty rcs are evaluated as 0
            for i, rc in enumerate(rc_list):
                sum = rc[0]
                count = rc[1]

                if count != 0:
                    rc_final[i] = sum / count
                else:
                    rc_final[i] = 0

        rc_final = np.array(rc_final)
        
        # print(f"speaker: {speaker}")
        # print(f"rc_final: {rc_final}")
        # print(f"rc_final[0] type: {type(rc_final[0])}")
        # print(f"rc_final lenth: {len(rc_final)}")
        # print(type(speaker_rcs))
        speaker_rcs.append((speaker, rc_final))

    return speaker_rcs
        
# Balance Labels
def balance_labels(dataset):

    label_0 = 0
    label_1 = 0

    for data in dataset:
        audio1, phoneme1, text1, speaker1, rcs1, audio2, phoneme2, text2, speaker2, rcs2, label = data

        if label == 0:
            label_0 += 1
        else:
            label_1 += 1
    
    balanced = []
    num_excluded = label_1 - label_0
    # print(f"label 0: {label_0}, label 1: {label_1}")
    # print(f"num_excluded: {num_excluded}")
    curr_excluded = 0

    # Shuffle the dataset
    dataset_list = [dataset[i] for i in range(len(dataset))]
    random.shuffle(dataset_list)

    for i in range(len(dataset_list)):
        item = dataset_list[i]

        if item[-1] == 0:
            balanced.append(item)
        elif item[-1] == 1 and curr_excluded < num_excluded:
            curr_excluded += 1
        else:
            balanced.append(item)
    
    return balanced

# Balance Labels for both train and test and save it
def balance_labels_mult(train_paired, test_paired, train_dst, test_dst):
    train_b = balance_labels(train_paired)
    test_b = balance_labels(test_paired)

    save_dataset(train_b, test_b, train_dst, test_dst)

# Balance Labels
def balance_labels_agg(dataset):

    label_0 = 0
    label_1 = 0

    for data in dataset:
        speaker1, rcs1, speaker2, rcs2, label = data

        if label == 0:
            label_0 += 1
        else:
            label_1 += 1
    
    balanced = []
    num_excluded = label_1 - label_0
    # print(f"label 0: {label_0}, label 1: {label_1}")
    # print(f"num_excluded: {num_excluded}")
    curr_excluded = 0

    # Shuffle the dataset
    dataset_list = [dataset[i] for i in range(len(dataset))]
    random.shuffle(dataset_list)

    for i in range(len(dataset_list)):
        item = dataset_list[i]

        if item[-1] == 0:
            balanced.append(item)
        elif item[-1] == 1 and curr_excluded < num_excluded:
            curr_excluded += 1
        else:
            balanced.append(item)
    
    return balanced

# Create RCSpairs for different aggregations
def balance_labels_agg_mult(train_tests_paired):

    for train_test in train_tests_paired:
        num_agg, train, test = train_test

        # balance
        train = balance_labels_agg(train)
        test = balance_labels_agg(test)

        train_dst = f"data/processed/train_agg_{num_agg}_b.pkl"
        test_dst = f"data/processed/test_agg_{num_agg}_b.pkl"

        save_dataset(train, test, train_dst, test_dst)

        print(f"After balancing for aggregation of {num_agg}")
        print(f"Train length: {len(train)}")
        print(f"Test length: {len(test)}")
    
    return train_tests_paired

def preprocess_single(rcs, epochs, sr, threshold_vc, num_tubes, vowels, offset, dst_train, dst_test):
    since = time.time()
    print("Calculating and adding RCS for Train dataset")
    dataset_train = AudioDataset("data/Train", rcs, epochs, sr, threshold_vc, num_tubes, vowels, offset)
    # dataset_train_f = AudioPair(dataset_train)
    with open(dst_train, 'wb') as f:
        pickle.dump(dataset_train, f)
        print("Saved Train_w_rcs_acc.pkl")
    print(f"Train dataset preprocessed in {time.time() - since}s")

    # print(f"dataset_single: {len(dataset_single)}")
    since = time.time()
    print("Calculating and adding RCS for Test dataset")
    dataset_test = AudioDataset("data/Test", rcs, epochs, sr, threshold_vc, num_tubes, vowels, offset)
    # dataset_test_f = AudioPair(dataset_test)
    print(f"Test dataset preprocessed in {time.time() - since}s")
    with open(dst_test, 'wb') as f:
        pickle.dump(dataset_test, f)
        print("Saved Test_w_rcs_acc.pkl")
    print(f"Test dataset preprocessed in {time.time() - since}s")
    
    print(f"Data preprocessing took {time.time() - since}s")

# Aggregated Datasets
def preprocess_agg(dataset_single_train, dataset_single_test, num_samples):
    dst_train = f'data/processed/train_agg_{num_samples}.pkl'
    dst_test = f'data/processed/test_agg_{num_samples}.pkl'

    speaker_rcs_train = into_full_phoneme(dataset_single_train, num_samples)
    speaker_rcs_test = into_full_phoneme(dataset_single_test, num_samples)
    print(len(speaker_rcs_train))
    print(len(speaker_rcs_test))

    with open(dst_train, 'wb') as f:
        pickle.dump(speaker_rcs_train, f)
        print(f"Saved as {dst_train}")
    
    with open(dst_test, 'wb') as f:
        pickle.dump(speaker_rcs_test, f)
        print(f"Saved as {dst_test}")

# Create aggregated datasets given list of aggregation counts
def preprocess_agg_it(dataset_single_train, dataset_single_test, num_aggs):
    for num_samples in num_aggs:
        print(f"For aggregating {num_samples} samples.")
        preprocess_agg(dataset_single_train, dataset_single_test, num_samples)
    