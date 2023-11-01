from utils import audio_seg, audio_visual, read_phoneme
from transfer import TF
from core import f_res_transfer
import numpy as np
from core import rcs_single

# Variables
SR = 16000
N = 8
RCS = np.zeros(N)
OFFSET = 0.01
EPOCHS = 1000
THRESHOLD_E = 0.001
L_TUBE = 17.5
V_SOUND = 35000
TAU = L_TUBE / (V_SOUND * N)
print(f"TAU: {TAU}")
print(f"TAU^-1: {1 / TAU}")

# Phoneme categories
stops = {"b", "d", "g", "p", "t", "k", "dx", "q"}
affricates = {"jh", "ch"}
fricatives = {"s", "sh", "z", "zh", "f", "th", "v", "dh"}
nasals = {"m", "n", "ng", "em", "en", "eng", "nx"}
semivowels_glides = {"l", "r", "w", "y", "hh", "hv", "el"}
vowels = {"iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"}
others = {"pau", "epi", "h#", "1", "2"}

# audio_visual("SA1.WAV.wav", "SA1.PHN", SR)

audio_segs = audio_seg("SA1.WAV.wav", read_phoneme("SA1.PHN"))

for seg in audio_segs:
    audio = seg[0]
    phoneme = seg[1]
    start = seg[2]
    end = seg[3]
    # print(f"phoneme: {phoneme}")
    # print(f"start: {start}")
    # print(f"end: {end}")

    if phoneme in vowels:
        print(f"For Phoneme: {phoneme}")
        rcs_single(OFFSET, audio, phoneme, RCS, EPOCHS, SR, THRESHOLD_E, N)