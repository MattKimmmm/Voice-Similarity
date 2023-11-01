# Voice-Similarity

Vocal Reconstruction Setup

Dataset: TIMIT with a sampling rate (SR) of 16 kHz

1. Frequency Response from Speech:
- For a given vowel segment from the TIMIT dataset, compute its frequency response using the FFT. This will act as the target frequency response.
2. Transfer Function Computation:
- Initialize the reflection coefficients. These coefficients will define the characteristics of the vocal tract for the given vowel.
- Using the initialized reflection coefficients, calculate the frequency response of the vocal tract model via a transfer function over a specified range of frequencies.
3. Optimization:
- Compare the target frequency response with the modeled frequency response.
- Adjust the reflection coefficients iteratively to minimize the difference between the two frequency responses.

Outcome: Approximated reflection coefficients of the vocal tract that should model the unique vowel phonemes.