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
                print(f"For speaker {speaker}")

                if not os.path.isdir(speaker_dir):
                    continue

                # from the directory name extract gender, initials, and index
                # print(f"Loading data from {speaker} speaker")
                gender = speaker[0]
                initials = speaker[1:3]
                index = speaker[4]

                files = sorted(os.listdir(speaker_dir))

                # Initialize Variables
                phoneme_f = None
                text_f = None
                wav_f = None
                
                for file in files:
                    if file.endswith(".PHN"):
                        phoneme_f = os.path.join(speaker_dir, file)
                    elif file.endswith(".TXT"):
                        text_f = os.path.join(speaker_dir, file)
                    elif file.endswith(".wav"):
                        wav_f = os.path.join(speaker_dir, file)

                    # If all files are found, append to list
                    if phoneme_f and text_f and wav_f:
                        since = time.time()
                        self.phonemes.append(phoneme_f)
                        self.texts.append(text_f)
                        self.audios.append(wav_f)
                        self.speakers.append(speaker)
                        # print(f"speaker: {speaker}")
                        # print(phoneme_f)
                        # print(text_f)
                        # print(wav_f)
                        # print("")
                        
                        # Extract rcs for the audio file and append to list
                        # print(phoneme_f)
                        self.rcs.append(audio_single_pp(self.rcs_init, self.epochs, self.sr, self.threshold_vc, self.num_tubes, wav_f, phoneme_f, self.vowels, self.offset))
                        print(f"Single audio took {time.time() - since}s")

                        # Reset Variables
                        phoneme_f = text_f = wav_f = None
                        count += 1
        
        print(f"Total of {count} sentences")
        
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

        audio1, phoneme1, text1, speaker1, rcs1 = self.audio_dataset[idx1]
        audio2, phoneme2, text2, speaker2, rcs2 = self.audio_dataset[idx2]

        gender1 = speaker1[0]
        initials1 = speaker1[1:3]
        index1 = speaker1[4]
        gender2 = speaker2[0]
        initials2 = speaker2[1:3]
        index2 = speaker2[4]

        label = 1
        if (initials1 == initials2) and (index1 == index2):
            label = 0
        
        return audio1, phoneme1, text1, speaker1, rcs1, audio2, phoneme2, text2, speaker2, rcs2, label
    

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
def into_full_phoneme(dataset):

    rcs_dict = {}
    phoneme_dict = {}
    text_dict = {}

    for data in dataset:
        audio1, phoneme1, text1, speaker1, rcs1, audio2, phoneme2, text2, speaker2, rcs2, label = data

        if speaker1 in rcs_dict.keys():
            rcs_dict[speaker1].append(rcs1)
        else:
            rcs_dict[speaker1] = []
            rcs_dict[speaker1].append(rcs1)
        
        if speaker1 in phoneme_dict.keys():
            phoneme_dict[speaker1].append(phoneme1)
        else:
            phoneme_dict[speaker1] = []
            phoneme_dict[speaker1].append(phoneme1)
        
        if speaker1 in text_dict.keys():
            text_dict[speaker1].append(text1)
        else:
            text_dict[speaker1] = []
            text_dict[speaker1].append(text1)

    # print(f"rcs_dict shape: {len(rcs_dict)}")
    # print(f"rcs_dict: {rcs_dict}")
    # print(f"phoneme_dict shape: {len(phoneme_dict)}")
    # print(f"phoneme_dict: {phoneme_dict}")
    # print(f"text_dict shape: {len(text_dict)}")
    # print(f"text_dict: {text_dict}")

    speaker_rcs = []

    # Aggregate RCS for each speaker
    for item in rcs_dict.items():
        speaker, rcs = item
        # print(f"Speaker: {speaker}")
        # print(f"RCS: {len(rcs[0])}")

        rc_list = [[0, 0] for _ in range(320)]
        rc_final = [0] * 320
        # print(rc_list)

        for rc_single in rcs:
            # print(f"rc_single length: {len(rc_single)}")
            # print(rc_single)

            for i, rc in enumerate(rc_single):
                # print(i)
                # print(rc)
                if rc != 0:
                    # print(f"current sum: {rc_list[i][0]}")
                    # print(f"current count: {rc_list[i][1]}")
                    rc_list[i][0] += rc
                    rc_list[i][1] += 1
                    # print("After update")
                    # print(f"current sum: {rc_list[i][0]}")
                    # print(f"current count: {rc_list[i][1]}")
            
            # print(f"rc_list length: {len(rc_list)}")
            # print(rc_list)
                
            for i, rc in enumerate(rc_list):
                sum = rc[0]
                count = rc[1]

                if count != 0:
                    rc_final[i] = sum / count
                else:
                    rc_final[i] = 0

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
def preprocess_agg(dst_train, dst_test, dataset_single_train, dataset_single_test):

    speaker_rcs_train = into_full_phoneme(dataset_single_train)
    speaker_rcs_test = into_full_phoneme(dataset_single_test)
    agg_train_single = RCSDataset(speaker_rcs_train)
    agg_test_single = RCSDataset(speaker_rcs_test)

    with open(dst_train, 'wb') as f:
        pickle.dump(agg_train_single, f)
        print("Saved agg_train.pkl")
    
    with open(dst_test, 'wb') as f:
        pickle.dump(agg_test_single, f)
        print("Saved agg_test.pkl")

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