import numpy as np
from numba import njit


# Calculate z from omega
@njit
def z_calc(omega, sr):
    z = np.exp(1j * 2 * np.pi * np.complex64(omega) / np.complex64(sr))
    # complex64
    return np.complex64(z)

# Numerator
@njit
def num(z_val, rcs, N):
    ans = np.complex64(1)
    for rc in rcs:
        ans = ans * (1 + rc)
    ans = ans * (z_val ** (- N / 2))
    # complex64
    return ans

# Denominator - D(z)
@njit
def den(z_val, rcs):
    ans = np.array([1, -1], dtype=np.complex64)
    for rc in rcs:
        addition = np.array([[1, -rc], [-rc * z_val ** (-1), z_val ** (-1)]], dtype=np.complex64)
        ans = np.dot(ans, addition)
    
    ans = np.dot(ans, np.array([[1], [0]], dtype=np.complex64))
    return ans

# Given an audio segment, corresponding phoneme label, and initial reflection coefficients. Return the transfer function.
@njit
def tf(rcs, omega, sr, N):

    z_val = z_calc(omega, sr)
    
    # rcs into complex64
    rcs = rcs.astype(np.complex64)
    numerator = num(z_val, rcs, N)
    # print(f"numerator_shape: {numerator.shape}")
    # print(f"numerator: {numerator}")
    denominator = den(z_val, rcs)
    # print(f"denominator_shape: {denominator.shape}")
    # print(f"denominator: {denominator}")
    # print(f"denominator dtype: {denominator.dtype}")
    f_res = np.abs(numerator / denominator)
    # print f_res dtype
    # print(f"f_res dtype: {f_res.item().dtype}")
    # print(f"f_res: {f_res}")
    # print(f"f_res.item(): {f_res.item()}")
    f_res = np.array(f_res.item(), dtype=np.float32)

    # val = 600 * np.log10(np.abs(f_res.item()))

    # float32
    return f_res