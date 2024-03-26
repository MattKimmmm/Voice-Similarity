import matplotlib.pyplot as plt
import os
from scipy import fftpack
import numpy as np

# Plot the magnitude spectrum given x and y
def plot_signal(x, y, path, title, phoneme, is_org):
    plt.plot(x, y)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    if is_org:
        figpath = f"org_{phoneme}.png"

        filepath = os.path.join(path, figpath)
        plt.savefig(filepath)
    else:
        figpath = f"tf_{phoneme}.png"
        filepath = os.path.join(path, figpath)
        plt.savefig(filepath)
        plt.clf()


# Plot ROC Curve
def plot_roc(fpr, tpr,roc_auc, margin, path):
    print("plot_roc")

    figpath = f"siamese_margin_{margin}"
    filepath = os.path.join(path, figpath)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filepath)

# Plot accuracy v number of audio aggregation
def plot_reg():
    print("plot_reg")
    x = [1, 2, 5, 10]
    y = [.6458, .8930, .9474, .9940]

    plt.figure()
    plt.plot(x, y)
    plt.ylim([0.0, 1.0])
    plt.xlabel('Number of Audio Aggregation')
    plt.ylabel('Accuracy')
    plt.savefig('figures/performance')
    plt.show()

def plot_roc_agg(fpr, tpr,roc_auc, margin, path, agg_num):
    print("plot_roc")

    figpath = f"agg_{agg_num}_siamese_margin_{margin}"
    filepath = os.path.join(path, figpath)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filepath)

# plot_roc_agg for model
def plot_roc_agg_model(fpr, tpr,roc_auc, margin, path, agg_num, model_name):
    print("plot_roc")

    figpath = f"agg_{agg_num}_model_{model_name}_siamese_margin_{margin}_balanced"
    filepath = os.path.join(path, figpath)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filepath)

def show_wav(audio, start, end, num_samples, path, sr):
    
    # Display the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(start, end, num=num_samples), audio)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    # plt.show()
    wave_path = os.path.join(path, "org_waveform.png")
    plt.savefig(wave_path)

    # FFT
    yf = fftpack.fft(audio)
    xf = np.linspace(0.0, 0.5 * sr, len(audio) // 2)  # frequencies

    # Plot the magnitude spectrum
    # plt.plot(xf, 2.0/len(audio) * np.abs(yf[0:len(audio)//2]))
    # dB not normalized
    plt.plot(xf, 20 * np.log10(np.abs(yf[0:len(audio)//2])))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')
    # plt.show()
    freq_path = os.path.join(path, "org_magnitude_spectrum.png")
    plt.savefig(freq_path)

    # From transfer function
    # f_ress = f_res_transfer()
    # xf = np.linspace(0.0, 0.5 * SR, len(f_ress))
    # plt.plot(xf, f_ress)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('Magnitude Spectrum')
    # plt.show()