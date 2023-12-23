from utils import audio_visual, stats, stats_agg
import numpy as np
from process_audio import rcs_single, audio_single, audio_single_paper, AudioDataset, AudioPair, into_full_phoneme
from process_audio import balance_labels, RCSDataset, RCSPair, balance_labels_agg
from siamese import SiameseNetwork, ContrastiveLoss, Siamese_dropout, Siamese_dropout_hidden, Siamese_st16, Siamese_st8
from siamese import Siamese_fc, Siamese_Conv, Siamese_Conv_fc
from test_siamese import test_loop
from tunes import margin_threshold_siamese, margin_threshold_siamese_agg
import pickle
import time

import torch
from torch.utils.data import DataLoader
from torch import optim

from process_audio import preprocess_single

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
THRESHOLD_VC = 0.001
BATCH_SIZE = 16
PRED_TRESHOLDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
MARGINS = [1, 2, 5, 10, 20, 30]
MARGIN = 1

SINGLE_TRAIN = 'data/processed/train_w_rcs.pkl'
SINGLE_TEST = 'data/processed/test_w_rcs.pkl'
AGG_TRAIN = 'data/processed/train_agg.pkl'
AGG_TEST = 'data/processed/test_agg.pkl'

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

# audio_visual("SA1.WAV.wav", "SA1.PHN", SR, vowels)

# results = audio_single(RCS, EPOCHS, SR, THRESHOLD_VC, N, "SA1.WAV.wav", "SA1.PHN", vowels, OFFSET)

def main():
    # Data Preprocessing
    # Extract RCS from audios and create datasets
    preprocess_single(RCS, EPOCHS, SR, THRESHOLD_VC, N, vowels, OFFSET, SINGLE_TRAIN, SINGLE_TEST)

    # dataset = AudioPair(dataset_single)

    # Load datasets
    # with open(SINGLE_PAIR_TRAIN, 'rb') as f:
    #     dataset_train_single = pickle.load(f)
    #     print("Loaded pair_rcs.pkl")
    #     print(f"Train Dataset length: {len(dataset_train_single)}")
        
    # with open(SINGLE_PAIR_TEST, 'rb') as f:
    #     dataset_test_single = pickle.load(f)
    #     print("Loaded pair_rcs.pkl")
    #     print(f"Test Dataset length: {len(dataset_test_single)}")

    

    # Create aggregated dataset
    # preprocess_agg(AGG_TRAIN, AGG_TEST, dataset_train_single, dataset_test_single)

    # with open(AGG_TRAIN, 'rb') as f:
    #     agg_train_single = pickle.load(f)
    #     print("Loaded agg_train.pkl")
    #     print(f"Train Dataset length: {len(agg_train_single)}")
    #     # for i in range(len(dataset)):
    #     #     print(f"dataset[{i}]: {dataset[i]}")
    # with open(AGG_TEST, 'rb') as f:
    #     agg_test_single = pickle.load(f)
    #     print("Loaded agg_test.pkl")
    #     print(f"Test Dataset length: {len(agg_test_single)}")

    # agg_train = RCSPair(agg_train_single)
    # agg_test = RCSPair(agg_test_single)

    # for i in range(len(agg_train)):
    #     # print(agg_train[i])
    #     print(len(agg_train[i]))

    # Anaylitics
    # stats(dataset_train_single, dataset_test_single)
    # stats_agg(agg_train, agg_test)

    # print(agg_train[0])

    # Truncate Datasets
    # dataset_train_single = balance_labels(dataset_train_single)
    # agg_train = balance_labels_agg(agg_train)
    # dataset_test_single = balance_labels(dataset_test_single)
    # agg_test = balance_labels_agg(agg_test)
    # print(f"Balanced Training Set length: {len(agg_train)}")
    # print(f"Balanced Test Set length: {len(agg_test)}")
    # print("agg_train: ", len(agg_train[0]))
    # print("agg_test: ", agg_test[0])
    # print(f"agg_train len: {len(agg_train)}")
    # print(f"agg_test len: {len(agg_test)}")

    # siamese = SiameseNetwork()
    # dataloader_train = DataLoader(dataset_train_single, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    # dataloader_test = DataLoader(dataset_test_single, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # for i in range(len(speaker_rcs)):
    #     speaker, rcs = speaker_rcs[i]
    #     print(f"For speaker: {speaker}")
    #     print(f"Aggregated RCS: {rcs}")

    # Train
    # train_loop(siamese, dataloader_train, ContrastiveLoss(margin=MARGIN), optim.Adam(siamese.parameters(), lr=0.0005), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, DEVICE, MARGIN)

    # Test
    # state_dict = torch.load("models/siamese_1204.pth")
    # siamese.load_state_dict(state_dict)
    # for pred_treshold in PRED_TRESHOLDS:
    #     print(f"pred_threshold: {pred_treshold}")
    #     test_loop(siamese, dataloader_test, ContrastiveLoss(margin=MARGIN), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, DEVICE, pred_treshold)
    #     print("")

    # For paper, single audio outputs
    # audio_single_paper(RCS, EPOCHS, SR, THRESHOLD_VC, N, "data/Train/DR4/FALR0/SA1.WAV.wav", "data/Train/DR4/FALR0/SA1.PHN", "data/Train/DR4/FALR0/SA1.TXT",vowels, OFFSET)
    
    # Hyperparameter Tuning
    # margin_threshold_siamese_agg(MARGINS, siamese, dataloader_train, dataloader_test, 
    #                          optim.Adam(siamese.parameters(), lr=0.0005), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET, 
                            #  DEVICE)

if __name__ == "__main__":
    main()
