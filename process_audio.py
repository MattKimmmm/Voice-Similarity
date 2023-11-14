from transfer import TF
import numpy as np
from scipy import fftpack
from utils import plot_signal, audio_seg, read_phoneme
import os
from itertools import combinations

import torch
from torch.utils.data import Dataset, DataLoader

# Custom Audio Dataset for TIMIT data
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.audios = []    # Audio file
        self.speakers = []  # Speaker ID
        self.phonemes = []  # Phoneme file
        self.texts = []     # Text file
        self.root_dir = root_dir

        self._load_data()

    def _load_data(self):
        # Create pair of audio files
        print("Loading data...")

        # Currently just DR1

        # for dialect in os.listdir(self.root_dir):
        #     # print(f"Loading data from {dialect} dialect")
        #     dialect_dir = os.path.join(self.root_dir, dialect)

        dialect_dir = os.path.join(self.root_dir, "DR1")

        # Check if its a directory (not .DS_Store)
        # if not os.path.isdir(dialect_dir):
        #     continue

        for speaker in os.listdir(dialect_dir):
            # Iterate over files in speaker directory and extract relevant files
            speaker_dir = os.path.join(dialect_dir, speaker)

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
                    self.phonemes.append(phoneme_f)
                    self.texts.append(text_f)
                    self.audios.append(wav_f)
                    self.speakers.append(speaker)
                    # print(f"speaker: {speaker}")
                    
                    # Reset Variables
                    phoneme_f = text_f = wav_f = None
    
    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio = self.audios[idx]
        phoneme = self.phonemes[idx]
        text = self.texts[idx]
        speaker = self.speakers[idx]

        return audio, phoneme, text, speaker
    
# Dataset that pairs audio data and labels the pair
# Same speaker = 0 / Different speaker = 1
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

        audio1, phoneme1, text1, speaker1 = self.audio_dataset[idx1]
        audio2, phoneme2, text2, speaker2 = self.audio_dataset[idx2]

        gender1 = speaker1[0]
        initials1 = speaker1[1:3]
        index1 = speaker1[4]
        gender2 = speaker2[0]
        initials2 = speaker2[1:3]
        index2 = speaker2[4]

        label = 1
        if (initials1 == initials2) and (index1 == index2):
            label = 0
        
        return audio1, phoneme1, text1, speaker1, audio2, phoneme2, text2, speaker2, label


# Calculate the frequency response of the input audio
def calc_fft(audio):
    num_samples = len(audio)
    f_res = fftpack.fft(audio)
    f_res = f_res[:num_samples // 2]
    f_res = 20 * np.log10(np.abs(f_res))
    # f_res = 20 * np.log10(f_res)
    f_res = f_res - np.mean(f_res)
    # f_res = np.abs(f_res)
    return f_res

# calculate the frequency response of the transfer function
def f_res_transfer(audio, phoneme, rcs, freqs, num_tubes):
    f_ress = []
    t_function = TF(audio, phoneme, rcs, freqs, num_tubes)

    for f in freqs:
        f_res = t_function.tf(f)
        f_ress.append(f_res)
    # print(f_ress)

    return 20 * np.log10(np.abs(f_ress))

# rcs training loop for a given phoneme
def rcs_single(offset, audio, phoneme, rcs, epochs, sr, threshold, num_tubes):
    # Variables
    loss_curr = 0
    loss_prev = float('inf')
    count = 0

    f_res_tf_up = []

    # frequency response from original audio = target
    # truncate it to the SR / 2
    num_samples = len(audio)
    f_res_org = calc_fft(audio)
    # print(f"len(f_res_org): {len(f_res_org)}")
    # print(f"f_res_org: {f_res_org}")

    # Calculate the corresponding frequency values for original audio
    freq_bin = np.arange(0, num_samples) * sr / num_samples
    freq_bin_pos = freq_bin[:num_samples // 2]
    # print("freq_bin len: ", len(freq_bin_pos))
    # print("freq_bin: ", freq_bin_pos)

    for i in range(epochs):
        # Iterate over rcs
        loss_tube = []
        for j in range(len(rcs)):
            # Add offset to rc
            rcs_up = rcs.copy()
            rcs_down = rcs.copy()
            rcs_up[j] += offset
            rcs_down[j] -= offset
            # print(rcs)

            # frequency response from transfer function for rc_up and rc_down
            f_res_tf_up = f_res_transfer(audio, phoneme, rcs_up, freq_bin_pos, num_tubes)
            f_res_tf_down = f_res_transfer(audio, phoneme, rcs_down, freq_bin_pos, num_tubes)
            # print(f"len(f_res_tf_up): {len(f_res_tf_up)}")

            # calculate loss
            loss_up = np.sum(np.abs(f_res_org - f_res_tf_up))
            loss_down = np.sum(np.abs(f_res_org - f_res_tf_down))
            # print(f"loss_up: {loss_up}")
            # print(f"loss_down: {loss_down}")

            # update rcs
            if loss_up < loss_down:
                rcs = rcs_up
                loss_curr = loss_up
            else:
                rcs = rcs_down
                loss_curr = loss_down
            
            loss_tube.append(loss_curr)
            

            # print("Current Offset at: ", offset)
        
        # loss avg over one whole tube
        loss_avg_tube = np.mean(loss_tube)

        if np.abs(loss_avg_tube - loss_prev) < threshold:
            print("Error improvement less than threshold. Terminating")
            break

        if loss_avg_tube > loss_prev:
            offset *= 0.5
            loss_prev = loss_curr
            count += 1

            if count > 4:
                print("Error not improving in 5 consecutive steps. Terminating")
                break
            
        # when loss < loss_prev
        if loss_avg_tube < loss_prev:
            offset *= 1.01
            loss_prev = loss_curr

            count = 0
        
        loss_prev = loss_avg_tube
        
        if i % 20 == 0:
            print(f"Loss at Epoch {i}: {loss_avg_tube}")
            print("Current rcs: ", rcs)

    print(f"Outputs for {phoneme}:")
    print(f"Final rcs: {rcs}")
    print(f"Final loss: {loss_avg_tube}")
    # plot f_res_org and f_res_tf_up
    path = "./figures/vocal_tract_1109"
    title = f"Frequency Response of {phoneme}"
    # plot_signal(freq_bin_pos, f_res_org, path, title, phoneme, True)
    title = f"V(z) of {phoneme}"
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

# Batch-modified
# Takes in a single audio file and phoneme segmentation file and returns the input layer for the network
# 16 * 20 (# vowels) = 320
def audio_single(rcs, epochs, sr, threshold_vc, num_tubes, audio_wav, phoneme_seg, vowels, offset):
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
                print(f"For Phoneme: {phoneme}")
                rcs, error = rcs_single(offset, audio, phoneme, rcs, epochs, sr, threshold_vc, num_tubes)
                results.append((phoneme, rcs, error))

        for result in results:
            phoneme, rcs, error = result
            print(f"phoneme: {phoneme}")
            print(f"rcs: {rcs}")
            print(f"error: {error}")
    
        rcs_layer = make_input(results, vowels)
        rcs_layers.append(rcs_layer)

    return rcs_layers