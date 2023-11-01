from transfer import TF
import numpy as np
from scipy import fftpack
from utils import plot_signal

# calculate the frequency response of the transfer function
def f_res_transfer(audio, phoneme, rcs, freqs, num_tubes):
    f_ress = []
    t_function = TF(audio, phoneme, rcs, freqs, num_tubes)

    for f in freqs:
        f_res = t_function.tf(f)
        f_ress.append(f_res)
    # print(f_ress)

    return f_ress

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
    f_res_org = fftpack.fft(audio)
    # print(f"len(f_res_org): {len(f_res_org)}")
    # print(f"f_res_org: {f_res_org}")
    f_res_org = f_res_org[:num_samples // 2]

    # Calculate the corresponding frequency values for original audio
    freq_bin = np.arange(0, num_samples) * sr / num_samples
    freq_bin_pos = freq_bin[:num_samples // 2]
    print("freq_bin len: ", len(freq_bin_pos))
    print("freq_bin: ", freq_bin_pos)

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
                if loss_up > threshold:
                    rcs = rcs_up
                    loss_curr = loss_up
                else:
                    break
            else:
                if loss_down > threshold:
                    rcs = rcs_down
                    loss_curr = loss_down
                else:
                    break
            
            loss_tube.append(loss_curr)
            

            # print("Current Offset at: ", offset)
        
        # loss avg over one whole tube
        loss_avg_tube = np.mean(loss_tube)

        print(f"Loss at Epoch {i}: {loss_avg_tube}")
        print("Current rcs: ", rcs)

        if loss_avg_tube > loss_prev:
            # offset *= 0.5
            # loss_prev = loss_curr
            count += 1

            if count > 4:
                print("Error not improving in 5 consecutive steps. Terminating")
                break
            
        # when loss < loss_prev
        if loss_avg_tube < loss_prev:
            # offset *= 1.01
            # loss_prev = loss_curr

            count = 0
        
        loss_prev = loss_avg_tube

        # plot f_res_org and f_res_tf_up
        # plot_signal(freq_bin_pos, 20 * np.log10(np.abs(f_res_org)))
        # plot_signal(freq_bin_pos, 20 * np.log10(np.abs(f_res_tf_up)))