import numpy as np

# Given an audio segment, corresponding phoneme label, and initial reflection coefficients. Return the transfer function.
class TF:
    def __init__(self, audio, phoneme, rcs=[], freqs=[], N=None, gender=None, step=None, sr=None):
        self.audio = audio
        self.phoneme = phoneme
        
        if len(freqs) > 0:
            if isinstance(freqs, list):
                self.freqs = np.array(freqs)
            else:
                self.freqs = freqs

        if len(rcs) > 0:
            if isinstance(rcs, list):
                self.rcs = np.array(rcs)
            else:
                self.rcs = rcs

        if N:
            self.N = N
        else:
            self.N = len(rcs)
        if step:
            self.step = step
        else:
            self.step = 0.1
        if sr:
            self.sr = sr
        else:
            self.sr = 16000

    # transfer function
    def tf(self, omega):

        # Calculate z from omega
        def z_calc():
            z = np.exp(1j * 2 * np.pi * omega / self.sr)
            return z

        # Numerator
        def num(z_val):
            ans = 1
            for rc in self.rcs:
                ans = ans * (1 + rc)
            ans = ans * (z_val ** (- self.N / 2))
            return ans
        
        # Denominator - D(z)
        def den(z_val):
            ans = [1, -1]
            for rc in self.rcs:
                addition = [[1, -rc], [-rc * z_val ** (-1), z_val ** (-1)]]
                ans = np.matmul(ans, addition)
            
            ans = np.matmul(ans, [[1], [0]])
            return ans

        z_val = z_calc()
        numerator = num(z_val)
        # print(f"numerator_shape: {numerator.shape}")
        # print(f"numerator: {numerator}")
        denominator = den(z_val)
        # print(f"denominator_shape: {denominator.shape}")
        # print(f"denominator: {denominator}")
        f_res = abs(numerator / denominator)
        # val = 600 * np.log10(np.abs(f_res.item()))
        val = f_res.item()

        return val