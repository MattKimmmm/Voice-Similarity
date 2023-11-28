from utils import audio_seg, audio_visual, read_phoneme
import numpy as np
from process_audio import rcs_single, audio_single, AudioDataset, AudioPair
from siamese import SiameseNetwork, ContrastiveLoss
from train_siamese import train_loop
from test_siamese import test_loop
import pickle
import time

import torch
from torch.utils.data import DataLoader
from torch import optim

# Variables
SR = np.float32(16000)
N = np.float32(16)
# RCS = np.zeros(np.int32(N), dtype=np.float32)
RCS = np.array([-0.7, 0.0, 0.4, -0.5, 0.5, -0.3, 0.4, 0.0, 0.4, 0.1, 0.3, 0.4, 0.1, -0.1, 0.2, 0.0], dtype=np.float32)
OFFSET = 0.01
EPOCHS = 1000
THRESHOLD_E = 0.1
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
    # Data Preprocessing
    dataset_file_train = 'data/processed/Train_w_rcs.pkl'
    dataset_file_test = 'data/processed/Test_w_rcs.pkl'

    since = time.time()
    print("Calculating and adding RCS for Train dataset")
    dataset_train = AudioDataset("data/Train", RCS, EPOCHS, SR, THRESHOLD_VC, N, vowels, OFFSET)
    with open(dataset_file_train, 'wb') as f:
        pickle.dump(dataset_file_train, f)
        print("Saved Train_w_rcs_acc.pkl")
    print(f"Train dataset preprocessed in {time.time() - since}s")

    # print(f"dataset_single: {len(dataset_single)}")
    since = time.time()
    print("Calculating and adding RCS for Test dataset")
    dataset_test = AudioDataset("data/Test", RCS, EPOCHS, SR, THRESHOLD_VC, N, vowels, OFFSET)
    print(f"Test dataset preprocessed in {time.time() - since}s")
    with open(dataset_file_test, 'wb') as f:
        pickle.dump(dataset_file_test, f)
        print("Saved Test_w_rcs_acc.pkl")
    print(f"Test dataset preprocessed in {time.time() - since}s")
    
    # print(f"Data preprocessing took {time.time() - since}s")

    # dataset = AudioPair(dataset_single)

    # with open(dataset_file, 'rb') as f:
    #     dataset = pickle.load(f)
    #     print("Loaded pair_rcs.pkl")

    # siamese = SiameseNetwork()
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # Train
    # train_loop(siamese, dataloader, ContrastiveLoss(), optim.Adam(siamese.parameters(), lr=0.0005), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, DEVICE)

    # Test
    # state_dict = torch.load("models/siamese_1115.pth")
    # siamese.load_state_dict(state_dict)
    # test_loop(siamese, dataloader, ContrastiveLoss(), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, DEVICE)

if __name__ == "__main__":
    main()
