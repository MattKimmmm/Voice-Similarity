from utils import audio_seg, audio_visual, read_phoneme
import numpy as np
from process_audio import rcs_single, audio_single, AudioDataset, AudioPair
from siamese import SiameseNetwork, ContrastiveLoss
from train_siamese import train_loop
import pickle
import time

import torch
from torch.utils.data import DataLoader
from torch import optim

# Variables
SR = np.float32(16000)
N = np.float32(16)
RCS = np.zeros(np.int32(N), dtype=np.float32)
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
BATCH_SIZE = 16

# CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

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

def main():
    # since = time.time()
    # dataset_single = AudioDataset("data/TEST", RCS, EPOCHS, SR, THRESHOLD_VC, N, vowels, OFFSET)
    # print(f"dataset_single: {len(dataset_single)}")
    # dataset = AudioPair(dataset_single)

    dataset_file = 'data/processed/pair_rcs.pkl'

    # with open(dataset_file, 'wb') as f:
    #     pickle.dump(dataset, f)
    #     print("Saved pair_rcs.pkl")
    
    # print(f"Data preprocessing took {time.time() - since}s")

    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
        print("Loaded pair_rcs.pkl")

    siamese = SiameseNetwork()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    train_loop(siamese, dataloader, ContrastiveLoss(), optim.Adam(siamese.parameters(), lr=0.0005), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, DEVICE)

if __name__ == "__main__":
    main()
