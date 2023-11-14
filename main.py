from utils import audio_seg, audio_visual, read_phoneme
from transfer import TF
import numpy as np
from process_audio import rcs_single, audio_single, AudioDataset, AudioPair
from siamese import SiameseNetwork, ContrastiveLoss
from train_siamese import train_loop

from torch.utils.data import DataLoader
from torch import optim

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
BATCH_SIZE = 32

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
    dataset_single = AudioDataset(root_dir="data/TEST")
    print(f"dataset_single: {len(dataset_single)}")
    dataset = AudioPair(dataset_single)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    siamese = SiameseNetwork()
    print(f"dataset: {len(dataset)}")

    train_loop(siamese, dataloader, ContrastiveLoss(), optim.Adam(siamese.parameters(), lr=0.0005), EPOCHS, RCS, SR, THRESHOLD_VC, N, vowels, OFFSET)

if __name__ == "__main__":
    main()
