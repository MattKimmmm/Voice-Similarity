from utils import audio_seg, audio_visual, read_phoneme
from transfer import TF
import numpy as np
from process_audio import rcs_single, audio_single, AudioDataset, AudioPair
from siamese import SiameseNetwork, ContrastiveLoss

# Variables
SR = 16000
N = 16
RCS = np.zeros(N)
OFFSET = 0.01
EPOCHS = 1000
THRESHOLD_E = 0.001
L_TUBE = 17.5
V_SOUND = 35000
TAU = L_TUBE / (V_SOUND * N)    # tau = T / 2 = 3.125e-5
                                # = L_TUBE / (V_SOUND * N)
                                # L_TUBE / V_SOUND = 5e-4
                                # N = 16
# print(f"TAU: {TAU}")
# print(f"TAU^-1: {1 / TAU}")
THRESHOLD_VC = 0.1

# Phoneme categories
stops = {"b", "d", "g", "p", "t", "k", "dx", "q"}
affricates = {"jh", "ch"}
fricatives = {"s", "sh", "z", "zh", "f", "th", "v", "dh"}
nasals = {"m", "n", "ng", "em", "en", "eng", "nx"}
semivowels_glides = {"l", "r", "w", "y", "hh", "hv", "el"}
vowels = {"iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"}
others = {"pau", "epi", "h#", "1", "2"}

# audio_visual("SA1.WAV.wav", "SA1.PHN", SR)

# results = audio_single(RCS, EPOCHS, SR, THRESHOLD_VC, N, "SA1.WAV.wav", "SA1.PHN", vowels, OFFSET)


dataset = AudioPair(AudioDataset(root_dir="TIMIT/TEST"))
# print(f"audio_len: {len(dataset)}")
# print(f"AudioDataset: {dataset.audios}")

