import scipy.io as sio
import numpy as np
import time
import pickle

from process_audio import AudioDataset, AudioPair, into_full_phoneme, RCSDataset, read_phoneme, audio_seg
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
def stats(train_data, test_data):
    phoneme_len = []
    vowel_len = []

    for data in train_data:
        audio1, phoneme1, text1, speaker1, rcs1, audio2, phoneme2, text2, speaker2, rcs2, label = data

        with open(text1, 'r') as f:
            text = f.read()
            start, end, *_ = text.split()

            len_samples = int(end) - int(start)
            phoneme_len.append(len_samples)
        
        with open(text2, 'r') as f:
            text = f.read()
            start, end, *_ = text.split()

            len_samples = int(end) - int(start)
            phoneme_len.append(len_samples)

        len_samples_1 = sum(1 for value in rcs1 if value != 0)
        len_samples_2 = sum(1 for value in rcs2 if value != 0)
        len_vowels_1 = len_samples_1 // 16
        len_vowels_2 = len_samples_2 // 16
        vowel_len.append(len_vowels_1)
        vowel_len.append(len_vowels_2)
    
    for data in test_data:
        audio1, phoneme1, text1, speaker1, rcs1, audio2, phoneme2, text2, speaker2, rcs2, label = data

        with open(text1, 'r') as f:
            text = f.read()
            start, end, *_ = text.split()

            len_samples = int(end) - int(start)
            phoneme_len.append(len_samples)
        
        with open(text2, 'r') as f:
            text = f.read()
            start, end, *_ = text.split()

            len_samples = int(end) - int(start)
            phoneme_len.append(len_samples)

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