from scipy import fftpack
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# from core import f_res_transfer

# variables
SR = 16000

# Phoneme categories
stops = {"b", "d", "g", "p", "t", "k", "dx", "q"}
affricates = {"jh", "ch"}
fricatives = {"s", "sh", "z", "zh", "f", "th", "v", "dh"}
nasals = {"m", "n", "ng", "em", "en", "eng", "nx"}
semivowels_glides = {"l", "r", "w", "y", "hh", "hv", "el"}
vowels = {"iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"}
others = {"pau", "epi", "h#", "1", "2"}

# Process the phoneme label file and return the array [# segments, 3], where the column is [start time, end time, phoneme]
def read_phoneme(phoneme_org):
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

def audio_visual(audio_wav, phoneme_org, SR):
    phoneme_segs = read_phoneme(phoneme_org)
    audio_segs = audio_seg(audio_wav, phoneme_segs, SR)

    for seg in audio_segs:
        phoneme = seg[1]

        if phoneme in vowels:
            audio = seg[0]
            start = float(seg[2]) / SR
            end = float(seg[3]) / SR
            num_samples = len(audio)

            print(f"phoneme: {phoneme}")
            print(f"start: {start}")
            print(f"end: {end}")
            show_wav(audio, start, end, num_samples)

def show_wav(audio, start, end, num_samples):
    
    # Display the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(start, end, num=num_samples), audio)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    # FFT
    yf = fftpack.fft(audio)
    xf = np.linspace(0.0, 0.5 * SR, len(audio) // 2)  # frequencies

    # Plot the magnitude spectrum
    # plt.plot(xf, 2.0/len(audio) * np.abs(yf[0:len(audio)//2]))
    # dB not normalized
    plt.plot(xf, 20 * np.log10(np.abs(yf[0:len(audio)//2])))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')
    plt.show()

    # From transfer function
    # f_ress = f_res_transfer()
    # xf = np.linspace(0.0, 0.5 * SR, len(f_ress))
    # plt.plot(xf, f_ress)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('Magnitude Spectrum')
    # plt.show()

# segment audio file given phoneme labels. Return the segmented audio, corresponding phoneme labels, and the start and end time of each segment
def audio_seg(audio_wav, phoneme_seg):
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

# Plot the magnitude spectrum given x and y
def plot_signal(x, y):
    plt.plot(x, y)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')
    plt.show()